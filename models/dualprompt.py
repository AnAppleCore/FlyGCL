import logging
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit
from models.l2p import Prompt

logger = logging.getLogger()


class DualPrompt(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (0, 1),
                 len_g_prompt   : int   = 5,
                 pos_e_prompt   : Iterable[int] = (2,3,4),
                 len_e_prompt   : int   = 20,
                 g_pool         : int   = 1,
                 e_pool         : int   = 10,
                 prompt_func    : str   = 'prompt_tuning',
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 lambd          : float = 1.0,
                 backbone_name  : str   = None,
                 **kwargs):
        super().__init__()

        self.lambd = lambd
        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True
        
        # Slice the eprompt
        self.num_pt_per_task = int(e_pool / task_num)

        self.e_pool = e_pool
        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.ones(1).view(1))
        
        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(
                g_pool, 1, self.g_length * self.len_g_prompt, self.backbone.num_features, 
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(
                e_pool, 1, self.e_length * self.len_e_prompt, self.backbone.num_features, 
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(
                g_pool, 1, 2 * self.g_length * self.len_g_prompt, self.backbone.num_features, 
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(
                e_pool, 1, 2 * self.e_length * self.len_e_prompt, self.backbone.num_features, 
                _batchwise_selection=False, _diversed_selection=False, kwargs=self.kwargs
                )
        else: raise ValueError('Unknown prompt_func: {}'.format(prompt_func))
        self.g_prompt.key = None

    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, self.e_length, self.len_e_prompt, C)
        g_prompt = g_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.e_length, self.len_e_prompt, C)

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
        g_prompt = g_prompt.contiguous().view(B, 2 * self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, 2 * self.e_length, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat(xk, (g_prompt[:, pos_g * 2 + 0]), dim = 1)
                xv = torch.cat(xv, (g_prompt[:, pos_g * 2 + 1]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat(xk, (e_prompt[:, pos_e * 2 + 0]), dim = 1)
                xv = torch.cat(xv, (e_prompt[:, pos_e * 2 + 1]), dim = 1)
            
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

    def forward(self, inputs : torch.Tensor, return_feat=False) :
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0]

        if self.g_prompt is not None:
            g_p = self.g_prompt.prompts[0]
            g_p = g_p.expand(B, -1, -1)
        else:
            g_p = None

        if self.e_prompt is not None:
            start_id = self.task_count * self.num_pt_per_task
            end_id = (self.task_count+1) * self.num_pt_per_task
            if self.training and start_id < self.e_pool:
                res_e = self.e_prompt(query, s=start_id, e=end_id)
            else:
                res_e = self.e_prompt(query)
            e_s, e_p = res_e

        else:
            e_p = None
            e_s = 0

        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]
        x = self.backbone.fc(cls_token)

        self.similarity = e_s.mean()
        
        if return_feat:
            return x, cls_token
        else:
            return x

    def get_e_prompt_count(self):
        return self.e_prompt.update()

    def process_task_count(self):
        self.task_count += 1

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target) + self.lambd * self.similarity