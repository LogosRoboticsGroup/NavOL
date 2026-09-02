# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from torch.distributions import Normal
import math
import copy
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory, NavDP_RGBD_Backbone, PositionalEncoding, LearnablePositionalEncoding, SinusoidalPosEmb
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.storage import NavDPDiffusionStorage


class NavDP_Policy_DPT(nn.Module):
    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 channels=3):
        super().__init__()
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim

        # self.rgbd_encoder = NavDP_RGBD_Backbone(image_size, token_dim, memory_size=memory_size)
        self.point_encoder = nn.Linear(3,self.token_dim)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = token_dim,
                                                        nhead = heads,
                                                        dim_feedforward = 4 * token_dim,
                                                        activation = 'gelu',
                                                        batch_first = True,
                                                        norm_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer = self.decoder_layer,
                                             num_layers = self.temporal_depth)
        self.input_embed = nn.Linear(3,token_dim)
        
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 4)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)

        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        tgt_mask = (torch.triu(torch.ones(predict_size, predict_size, dtype=torch.float32)) == 1).transpose(0, 1)
        self.register_buffer("tgt_mask", tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0)), persistent=False)
        cond_critic_mask = torch.zeros((predict_size,4 + memory_size * 16), dtype=torch.float32)
        cond_critic_mask[:,0:4] = float('-inf')
        self.register_buffer("cond_critic_mask", cond_critic_mask, persistent=False)

    def predict_noise(self,last_actions,timestep,goal_embed,rgbd_embed):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep).unsqueeze(1)
        if time_embeds.shape[0] == 1 and last_actions.shape[0] > 1:
            time_embeds = time_embeds.tile((last_actions.shape[0],1,1))
        cond_embedding = torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask)
        output = self.layernorm(output)
        output = self.action_head(output)
        return output
    
    def predict_mix_noise(self,last_actions,timestep,goal_embeds,rgbd_embed):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep).unsqueeze(1)
        if time_embeds.shape[0] == 1 and last_actions.shape[0] > 1:
            time_embeds = time_embeds.tile((last_actions.shape[0],1,1))
        cond_embedding = torch.cat([time_embeds,goal_embeds[0],goal_embeds[1],goal_embeds[2],rgbd_embed],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embeds[0],goal_embeds[1],goal_embeds[2],rgbd_embed],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask)
        output = self.layernorm(output)
        output = self.action_head(output)
        return output
    
    def predict_critic(self,predict_trajectory,rgbd_embed):
        nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed],dim=1) +  self.cond_pos_embed(torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed],dim=1))
        critic_output = self.decoder(tgt = action_embeddings, memory = cond_embeddings, memory_mask = self.cond_critic_mask)
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:,0]
        return critic_output
    
class Timing:
    """
    From https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/svox2/utils.py#L611
    
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")
class NavdpPolicy(nn.Module):
    is_recurrent = False

    def __init__(
        self,        
        env_num=1,
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        channels=3,
        stop_threshold=-2.0,
        ft_denoising_steps=3,
        pretrained_model_path=None,
        task_name="nogoal",
        use_critic=True,
        **kwargs,
    ):
        if kwargs:
            print(
                "navdp_policy.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__()

        self.predict_size = predict_size
        self.image_size = image_size
        self.stop_threshold = stop_threshold
        self.memory_size = memory_size
        self.token_dim = token_dim
        self.ft_denoising_steps = ft_denoising_steps
        self.task = task_name
        self.use_critic = use_critic
        self.navi_former = NavDP_Policy_DPT(image_size=image_size,
                                             memory_size=memory_size,
                                             predict_size=predict_size,
                                             temporal_depth=temporal_depth,
                                             heads=heads,
                                             token_dim=token_dim)
        self.state_dict_not_load = {}
        if pretrained_model_path is not None and pretrained_model_path != "None":
            print(f"Loading pretrained model from {pretrained_model_path}")
            pretrained_state_dict = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
            state_dict = {}
            for key, value in pretrained_state_dict.items():
                if key.startswith("image_encoder.") or key.startswith("rgbd_encoder.") or key.startswith("pixel_encoder."):
                    self.state_dict_not_load[key] = value
                else:
                    state_dict[key] = value
            missing_keys, unexpected_keys = self.navi_former.load_state_dict(state_dict, strict=False)
            
            for name, param in self.navi_former.named_parameters():
                if 'point_encoder' in name:
                    param.requires_grad = False

            print("Missing keys:",missing_keys)
            print("Unexpected keys:",unexpected_keys)
        
        else:
            print("Training NavDP Policy from scratch")
            
        for name, param in self.navi_former.named_parameters():
            if 'point_encoder' in name:
                param.requires_grad = False

        print(f"Number of finetuned parameters: {sum(p.numel() for p in self.navi_former.parameters() if p.requires_grad)}")

        self.use_ddim = False
        self.denoising_steps = 10
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.denoising_steps,
                                    beta_schedule='squaredcos_cap_v2',
                                    clip_sample=True,
                                    prediction_type='epsilon')
        self.noise_scheduler.set_timesteps(self.denoising_steps)
        self.register_buffer("timesteps", self.noise_scheduler.timesteps, persistent=False)

        # Compile hot-path methods for kernel fusion and graph optimization.
        # NOTE: Cannot use mode="reduce-overhead" (CUDA graphs) because
        # predict_noise is first called under torch.inference_mode() during
        # rollout, then called again with gradients during training update.
        # CUDA graphs captured under inference_mode create inference tensors
        # that cannot be inplace-updated outside inference_mode.
        self.navi_former.predict_noise = torch.compile(self.navi_former.predict_noise)
        self.navi_former.predict_critic = torch.compile(self.navi_former.predict_critic)
        self.navi_former.predict_mix_noise = torch.compile(self.navi_former.predict_mix_noise)
        
    def act(self, observations):
        B = observations.shape[0]
        rgbd_embed_size = self.memory_size * 16 * self.token_dim
        rgbd_embed = observations[:, :rgbd_embed_size].reshape(B, self.memory_size * 16, self.token_dim) 
        device = rgbd_embed.device
        if self.task == "pointgoal":
            goal_embed = self.navi_former.point_encoder(observations[:, rgbd_embed_size:]).unsqueeze(1)
        elif self.task == "imagegoal":
            goal_embed = observations[:, rgbd_embed_size:]
        elif self.task == "nogoal":
            goal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        else:
            raise NotImplementedError
        
        noisy_action = torch.randn((B, self.predict_size, 3), device=device)
        for k in self.timesteps:
            noise_pred = self.navi_former.predict_noise(noisy_action, k.unsqueeze(0), goal_embed, rgbd_embed)
            noisy_action = self.noise_scheduler.step(model_output=noise_pred, timestep=k.item(), sample=noisy_action).prev_sample
        return noisy_action.reshape(B, -1)

    def act_inference(self, observations):
        if not self.use_critic:
            return self.act(observations)
        B = observations.shape[0]
        rgbd_embed_size = self.memory_size * 16 * self.token_dim
        rgbd_embed = observations[:, :rgbd_embed_size].reshape(B, self.memory_size * 16, self.token_dim) 
        device = rgbd_embed.device
        if self.task == "pointgoal":
            goal_embed = self.navi_former.point_encoder(observations[:, rgbd_embed_size:]).unsqueeze(1)
        elif self.task == "imagegoal":
            goal_embed = observations[:, rgbd_embed_size:]
        elif self.task == "nogoal":
            goal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        else:
            raise NotImplementedError
        
        sample_num = 16
        rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
        goal_embed = torch.repeat_interleave(goal_embed,sample_num,dim=0)
        noisy_action = torch.randn((B*sample_num, self.predict_size, 3), device=device)
        for k in self.timesteps:
            noise_pred = self.navi_former.predict_noise(noisy_action, k.unsqueeze(0), goal_embed, rgbd_embed)
            noisy_action = self.noise_scheduler.step(model_output=noise_pred, timestep=k.item(), sample=noisy_action).prev_sample
        
        self.all_trajectory = torch.cumsum(noisy_action / 4.0, dim=1).reshape(B, sample_num, self.predict_size, 3)[..., :2]
        
        critic_values = self.navi_former.predict_critic(noisy_action, rgbd_embed)
        critic_values = critic_values.reshape(B, sample_num)
        self.critic_values = critic_values
            
        sorted_indices = (-critic_values).argsort(dim=1)
        top1_indice = sorted_indices[:,0]
        batch_indices = torch.arange(B, device=device)
        noisy_action = noisy_action.reshape(B, sample_num, self.predict_size, 3)[batch_indices, top1_indice]
        
        return noisy_action.reshape(B, -1)

    def state_dict(self):
        state_dict = self.navi_former.state_dict()
        state_dict.update(self.state_dict_not_load)
        return state_dict
    
    def load_state_dict(self, state_dict):
        state_dict_new = {}
        for key, value in state_dict.items():
            if key.startswith("image_encoder.") or key.startswith("rgbd_encoder.") or key.startswith("pixel_encoder."):
                pass
            else:
                state_dict_new[key] = value
        missing_keys, unexpected_keys = self.navi_former.load_state_dict(state_dict_new, strict=False)
        return missing_keys, unexpected_keys
    
    def reset(self, dones=None):
        pass

    def compute_loss(self, observations, action, critic_values, is_valids):
        B = observations.shape[0]
        device = observations.device
        rgbd_embed_size = self.memory_size * 16 * self.token_dim
        rgbd_embed = observations[:, :rgbd_embed_size].reshape(B, self.memory_size * 16, self.token_dim) 
        if self.task == "pointgoal":
            goal_embed = self.navi_former.point_encoder(observations[:, rgbd_embed_size:]).unsqueeze(1)
        elif self.task == "imagegoal":
            goal_embed = observations[:, rgbd_embed_size:]
        elif self.task == "nogoal":
            goal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        else:
            raise NotImplementedError
        
        action = action.reshape(B, self.predict_size, 3)
        noise = torch.randn_like(action)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,(B,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        
        pred_noise = self.navi_former.predict_noise(noisy_action, timesteps, goal_embed, rgbd_embed)
        loss = F.mse_loss(pred_noise[is_valids], noise[is_valids])
        
        pred_critic_values = self.navi_former.predict_critic(action, rgbd_embed)
        critic_loss = F.mse_loss(pred_critic_values[is_valids], critic_values[is_valids])
        
        return loss, critic_loss