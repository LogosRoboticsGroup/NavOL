# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory, NavDP_RGBD_Backbone, PositionalEncoding, LearnablePositionalEncoding, SinusoidalPosEmb
from rsl_rl.utils import resolve_nn_activation


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
        
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 2)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)

        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=10,
                                       beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True,
                                       prediction_type='epsilon')
        tgt_mask = (torch.triu(torch.ones(predict_size, predict_size, dtype=torch.float32)) == 1).transpose(0, 1)
        self.register_buffer("tgt_mask", tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0)), persistent=False)
        cond_critic_mask = torch.zeros((predict_size,2 + memory_size * 16), dtype=torch.float32)
        cond_critic_mask[:,0:2] = float('-inf')
        self.register_buffer("cond_critic_mask", cond_critic_mask, persistent=False)
    
    def predict_noise(self,last_actions,timestep,goal_embed,rgbd_embed):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep).unsqueeze(1).tile((last_actions.shape[0],1,1))
        cond_embedding = torch.cat([time_embeds,goal_embed,rgbd_embed],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embed,rgbd_embed],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask)
        output = self.layernorm(output)
        output = self.action_head(output)
        return output
    
    def predict_critic(self,predict_trajectory,rgbd_embed):
        nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat([nogoal_embed,nogoal_embed,rgbd_embed],dim=1) +  self.cond_pos_embed(torch.cat([nogoal_embed,nogoal_embed,rgbd_embed],dim=1))
        critic_output = self.decoder(tgt = action_embeddings, memory = cond_embeddings, memory_mask = self.cond_critic_mask)
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:,0]
        return critic_output
    
    def predict_pointgoal_action(self, goal_point, rgbd_embed, sample_num=16):
        B = goal_point.shape[0]
        device = rgbd_embed.device
        with torch.no_grad():
            pointgoal_embed = self.point_encoder(goal_point).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed, sample_num, dim=0)

            naction = torch.randn((sample_num * B, self.predict_size, 3), device=device)
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.unsqueeze(0).to(device), pointgoal_embed, rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(B, sample_num)

            actions = naction.reshape(B,sample_num,self.predict_size,3) / 4.0
            return actions, critic_values

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(B,sample_num,self.predict_size,3)

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory, critic_values, positive_trajectory, negative_trajectory
    
    def predict_nogoal_action(self, rgbd_embed, sample_num=16):
        device = rgbd_embed.device
        B = rgbd_embed.shape[0]
        with torch.no_grad():
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
            nogoal_embed = torch.repeat_interleave(nogoal_embed,sample_num,dim=0)
           
            naction = torch.randn((sample_num * B, self.predict_size, 3), device=device)
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0).to(device), nogoal_embed, rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(B, sample_num)

            actions = naction.reshape(B,sample_num,self.predict_size,3) / 4.0
            return actions, critic_values
        
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(B, sample_num, self.predict_size, 3)

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:, 0:2]
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()


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
        

class DiffusionPolicy(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        channels=3,
        stop_threshold=-2.0,
                 
        pretrained_model_path="ckpt/navdp-weights.ckpt",
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "DiffusionPolicy.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

        self.predict_size = predict_size
        self.image_size = image_size
        self.stop_threshold = stop_threshold
        self.memory_size = memory_size
        self.token_dim = token_dim
        self.navi_former = NavDP_Policy_DPT(image_size=image_size,
                                             memory_size=memory_size,
                                             predict_size=predict_size,
                                             temporal_depth=temporal_depth,
                                             heads=heads,
                                             token_dim=token_dim)
        if pretrained_model_path is not None:
            print(f"Loading pretrained model from {pretrained_model_path}")
            self.navi_former.load_state_dict(torch.load(pretrained_model_path, map_location='cpu', weights_only=True), strict=False)
        self.memory_queue = None
    
    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)
        
        self.memory_queue = None

    def act(self, observations, hidden_states):
        B = observations.shape[0]
        device = observations.device
        images = torch.randn(B, self.image_size, self.image_size, 3, device=device)
        depths = torch.randn(B, self.image_size, self.image_size, 1, device=device)
        goals = torch.rand(B, 3, device=device) * 10 - 5  # Random goal in range [-5, 5]
        
        goals.clamp_(-10, 10)
        goals[:, 0].clamp_min_(0)
        
        actions, critic_values = self.navi_former.predict_pointgoal_action(goals, hidden_states, depths, sample_num=1)
        self.critic_values = critic_values
        return actions.reshape(B, -1)

    def act_inference(self, observations):
        # with Timing("DiffusionPolicy act_inference"):
        B = observations.shape[0]
        device = observations.device
        # images = torch.randn(B, self.image_size, self.image_size, 3, device=device)
        # depths = torch.randn(B, self.image_size, self.image_size, 1, device=device)

        rgbd_embed_size = (self.memory_size * 16) * self.token_dim
        rgbd_embed = observations[:, :rgbd_embed_size].reshape(B, self.memory_size * 16, self.token_dim)
        goals = observations[:, rgbd_embed_size:rgbd_embed_size+3]
        goals.clamp_(-10, 10)
        goals[:, 0].clamp_min_(0)
        
        #goals = torch.rand(B, 3, device=device) * 10 - 5  # Random goal in range [-5, 5]

        # if self.memory_queue is None:
        #     self.memory_queue = torch.zeros((B, self.memory_size, self.image_size, self.image_size, 3), device=device)
        # self.memory_queue = torch.cat((self.memory_queue[:, 1:], images[:, None]), dim=1)
        
        actions, critic_values = self.navi_former.predict_pointgoal_action(goals, rgbd_embed, sample_num=1)
        self.critic_values = critic_values
        return actions.reshape(B, -1) / 0.2

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        return self.critic_values

    def get_hidden_states(self):
        return self.memory_queue
