import logging
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

logger = logging.getLogger()


class MVP(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (0,1),
                 len_g_prompt   : int   = 5 ,
                 pos_e_prompt   : Iterable[int] = (2,3,4),
                 len_e_prompt   : int   = 20,
                 g_pool         : int   = 1,
                 e_pool         : int   = 10,
                 selection_size : int   = 1,
                 prompt_func    : str   = 'prompt_tuning',
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 lambd          : float = 1.0,
                 use_mask       : bool  = True,
                 use_contrastiv : bool  = True,
                 use_last_layer : bool  = False,
                 backbone_name  : str   = None,
                 **kwargs):

        super().__init__()

        self.lambd       = lambd
        self.kwargs      = kwargs
        self.task_num    = task_num
        self.use_mask    = use_mask
        self.num_classes = num_classes
        self.use_contrastiv  = use_contrastiv
        self.use_last_layer  = use_last_layer
        self.selection_size  = selection_size

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # Prompt
        self.g_pool = g_pool
        self.e_pool = e_pool = task_num
        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.zeros(1))

        self.register_buffer('count', torch.zeros(e_pool))
        self.learnable_key  = nn.Parameter(torch.randn(e_pool, self.backbone.embed_dim))
        self.learnable_mask = nn.Parameter(torch.zeros(e_pool, self.num_classes) - 1)

        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_size = 1 * self.g_length * self.len_g_prompt
            self.e_size = 1 * self.e_length * self.len_e_prompt
            self.g_prompts = nn.Parameter(torch.randn(g_pool, self.g_size, self.backbone.embed_dim))
            self.e_prompts = nn.Parameter(torch.randn(e_pool, self.e_size, self.backbone.embed_dim))

        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_size = 2 * self.g_length * self.len_g_prompt
            self.e_size = 2 * self.e_length * self.len_e_prompt
            self.g_prompts = nn.Parameter(torch.randn(g_pool, self.g_size, self.backbone.embed_dim))
            self.e_prompts = nn.Parameter(torch.randn(e_pool, self.e_size, self.backbone.embed_dim))
    
    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, -1, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, -1, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e]), dim = 1)
            x = block(x)
            x = x[:, :N, :]
        return x

    def prefix_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, -1, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, -1, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat((xk, g_prompt[:, pos_g * 2 + 0].clone()), dim = 1)
                xv = torch.cat((xv, g_prompt[:, pos_g * 2 + 1].clone()), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat((xk, e_prompt[:, pos_e * 2 + 0].clone()), dim = 1)
                xv = torch.cat((xv, e_prompt[:, pos_e * 2 + 1].clone()), dim = 1)
            
            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias
            
            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path1(block.ls1(attention))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def forward_features(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = x.clone()
            for n, block in enumerate(self.backbone.blocks):
                if n == len(self.backbone.blocks) - 1 and not self.use_last_layer: break
                query = block(query)
            query = query[:, 0]

        distance = 1 - F.cosine_similarity(query.unsqueeze(1), self.learnable_key, dim=-1)
        if self.use_contrastiv:
            mass = (self.count + 1)
        else:
            mass = 1.
        scaled_distance = distance * mass
        topk = scaled_distance.topk(self.selection_size, dim=1, largest=False)[1]
        distance = distance[torch.arange(topk.size(0), device=topk.device).unsqueeze(1).repeat(1,self.selection_size), topk].squeeze().clone()
        e_prompts = self.e_prompts[topk].squeeze().clone()
        mask = self.learnable_mask[topk].mean(1).squeeze().clone()
        
        if self.use_contrastiv:
            key_wise_distance = 1 - F.cosine_similarity(self.learnable_key.unsqueeze(1), self.learnable_key, dim=-1)
            self.similarity_loss = -((key_wise_distance[topk] / mass[topk]).exp().mean() / ((distance / mass[topk]).exp().mean() + (key_wise_distance[topk] / mass[topk]).exp().mean()) + 1e-6).log()
        else:
            self.similarity_loss = distance.mean()

        g_prompts = self.g_prompts[0].repeat(B, 1, 1)
        if self.training:
            with torch.no_grad():
                num = topk.view(-1).bincount(minlength=self.e_prompts.size(0))
                self.count += num

        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_prompts, e_prompts)
        feature = self.backbone.norm(x)[:, 0]
        mask = torch.sigmoid(mask)*2.
        return feature, mask

    def forward_head(self, feature : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.fc_norm(feature)
        x = self.backbone.fc(x)
        return x

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x, mask = self.forward_features(inputs, **kwargs)
        x = self.forward_head(x, **kwargs)
        if self.use_mask:
            x = x * mask
        return x

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target) + self.similarity_loss

    def get_similarity_loss(self):
        return self.similarity_loss

    def get_e_prompt_count(self):
        return self.count
    
    def process_task_count(self):
        self.task_count += 1