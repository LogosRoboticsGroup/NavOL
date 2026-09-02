import math
import torch
import torch.nn as nn
from .depth_anything.dpt import DepthAnythingV2


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # (seq_len,)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        position_encoding = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        return position_encoding


class NavDP_RGBD_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 memory_size=8):
        super().__init__()
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.rgb_model.eval()
        self.preprocess_mean = torch.tensor([0.485,0.456,0.406],dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229,0.224,0.225],dtype=torch.float32)
        
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.eval()
        self.former_query = LearnablePositionalEncoding(384,self.memory_size*16)
        self.former_pe = LearnablePositionalEncoding(384,(self.memory_size+1)*256)
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384,8,batch_first=True),2)
        self.project_layer = nn.Linear(384,embed_size)
        self.memory_queue = None
        self.zero_image_token = None
        
    def forward(self,images,depths):
        # device = images.device
        device = 'cuda'
        with torch.no_grad():
            if len(images.shape) == 4:
                if isinstance(images, torch.Tensor):
                    tensor_images = images
                else:
                    tensor_images = torch.as_tensor(images,dtype=torch.float32,device=device).permute(0,3,1,2)
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(device))/self.preprocess_std.reshape(1,3,1,1).to(device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
            elif len(images.shape) == 5:
                if isinstance(images, torch.Tensor):
                    tensor_images = images
                else:
                    tensor_images = torch.as_tensor(images,dtype=torch.float32,device=device).permute(0,1,4,2,3)
                B,T,C,H,W = tensor_images.shape
                tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(device))/self.preprocess_std.reshape(1,3,1,1).to(device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B,T*256,-1)
            if len(depths.shape) == 4:
                if isinstance(depths, torch.Tensor):
                    tensor_depths = depths
                else:
                    tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=device).permute(0,3,1,2)
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            elif len(depths.shape) == 5:
                if isinstance(depths, torch.Tensor):
                    tensor_depths = depths
                else:
                    tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=device).permute(0,1,4,2,3)
                B,T,C,H,W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B,T*256,-1)
            original_image_token = image_token.clone()
            former_token = torch.concat((image_token,depth_token),dim=1) + self.former_pe(torch.concat((image_token,depth_token),dim=1))
            former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size * 16, 384), device=device))
            memory_token = self.former_net(former_query,former_token)
            memory_token = self.project_layer(memory_token)
            return original_image_token, memory_token

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is not None:
            self.memory_queue[env_ids] = self.zero_image_token[:, None].repeat(env_ids.shape[0], self.memory_size, 1, 1)
        else:
            self.memory_queue = self.zero_image_token[:, None].repeat(self.memory_queue.shape[0], self.memory_size, 1, 1)

    def get_rgbd_token(self, images, depths):
        if self.zero_image_token is None:
            self.preprocess_mean = self.preprocess_mean.to(images.device)
            self.preprocess_std = self.preprocess_std.to(images.device)
            zero_image = (torch.zeros(1, 3, 224, 224, device=images.device) - self.preprocess_mean.reshape(1, 3, 1, 1)) / self.preprocess_std.reshape(1, 3, 1, 1)
            self.zero_image_token = self.rgb_model.get_intermediate_layers(zero_image)[0]

        B = images.shape[0]
        device = images.device
        with torch.no_grad():
            tensor_norm_images = (images - self.preprocess_mean.reshape(1, 3, 1, 1).to(device)) / self.preprocess_std.reshape(1, 3, 1, 1).to(device)
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]

            if self.memory_queue is None:
                self.memory_queue = self.zero_image_token[:, None].repeat(B, self.memory_size, 1, 1)
            self.memory_queue = torch.cat((self.memory_queue[:, 1:], image_token[:, None]), dim=1)
            
            image_token_queue = self.memory_queue.reshape(B, -1, 384)
            depth_token = self.depth_model.get_intermediate_layers(depths.repeat(1, 3, 1, 1))[0]

            tokens = torch.concat((image_token_queue, depth_token), dim=1)
            former_token = tokens + self.former_pe(tokens)
            former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size * 16, 384), device=device))
            memory_token = self.former_net(former_query, former_token)
            memory_token = self.project_layer(memory_token)
            return memory_token
        
class NavDP_ImageGoal_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512):
        super().__init__()
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.imagegoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.imagegoal_encoder = self.imagegoal_encoder.pretrained.float()
        self.imagegoal_encoder.patch_embed.proj = nn.Conv2d(in_channels=6,
                                                            out_channels = self.imagegoal_encoder.patch_embed.proj.out_channels,
                                                            kernel_size = self.imagegoal_encoder.patch_embed.proj.kernel_size,
                                                            stride = self.imagegoal_encoder.patch_embed.proj.stride,
                                                            padding = self.imagegoal_encoder.patch_embed.proj.padding)
        self.imagegoal_encoder.eval()
        self.project_layer = nn.Linear(384,embed_size)
        
    def forward(self,images):
        with torch.no_grad():
            assert len(images.shape) == 4 # B,C,H,W
            image_token = self.imagegoal_encoder.get_intermediate_layers(images)[0].mean(dim=1)
            image_token = self.project_layer(image_token)
            return image_token