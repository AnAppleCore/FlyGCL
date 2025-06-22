import logging
from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

logger = logging.getLogger()


class EnsemblePrompt(nn.Module):
    def __init__(self,
                 pool_num        : int,
                 pool_size       : int,
                 prompt_len      : int,
                 selection_size  : int,
                 feature_dim     : int,
                 expansion_dim   : int,
                 gating_func     : str = 'wta',
                 sparsity        : float = 0.05,
                 ema_weight      : float = 0.9,
                 **kwargs):
        super().__init__()

        self.pool_num        = pool_num
        self.pool_size       = pool_size
        self.prompt_len      = prompt_len
        self.selection_size  = selection_size
        self.feature_dim     = feature_dim
        self.expansion_dim   = expansion_dim
        self.gating_func     = gating_func
        self.sparsity        = sparsity
        self.ema_weight      = ema_weight

        self.use_rp          = expansion_dim > 0
        if not self.use_rp:
            expansion_dim    = feature_dim
        self.num_active      = max(1, int(expansion_dim * sparsity))  # Ensure at least 1

        self.key = nn.Parameter(torch.randn(pool_size, expansion_dim, requires_grad=True))
        torch.nn.init.uniform_(self.key, -1, 1)

        self.prompts = nn.ParameterList([])
        
        # Initialize first pool with random values (this is the trainable pool)
        first_prompt = nn.Parameter(torch.randn(pool_size, prompt_len, feature_dim, requires_grad=True))
        torch.nn.init.uniform_(first_prompt, -1, 1)
        self.prompts.append(first_prompt)
        
        # Initialize remaining pools as clones of the first pool (non-trainable)
        for i in range(1, pool_num):
            prompt = nn.Parameter(first_prompt.clone().detach(), requires_grad=False)
            self.prompts.append(prompt)

        if self.use_rp:
            self.random_projection = nn.Parameter(torch.randn(feature_dim, expansion_dim))
            # self.map_to_expert = nn.Parameter(torch.randn(expansion_dim, pool_size))
            self.random_projection.requires_grad = False
            # self.map_to_expert.requires_grad = False
        else:
            self.random_projection = None

        if gating_func == 'relu':
            self.non_linear_func = nn.ReLU()
        elif gating_func == 'sigmoid':
            self.non_linear_func = nn.Sigmoid()
        elif gating_func == 'tanh':
            self.non_linear_func = nn.Tanh()
        elif gating_func == 'softmax':
            self.non_linear_func = nn.Softmax(dim=-1)
        elif gating_func == 'gelu':
            self.non_linear_func = nn.GELU()
        elif gating_func == 'wta':
            self.non_linear_func = nn.Identity()
        elif gating_func == 'none':
            self.non_linear_func = nn.Identity()
        else:
            raise ValueError(f"Invalid gating function: {gating_func}")
        
    def init_prompts(self):
        first_prompt = self.prompts[0]
        for i in range(1, self.pool_num):
            self.prompts[i].data.copy_(first_prompt.data)
            self.prompts[i].requires_grad = False
        self.prompts[0].requires_grad = True

    @torch.no_grad()
    def ema_update_prompts(self, source: int, target: int, ema_weight: float=None):
        if ema_weight is None:
            ema_weight = self.ema_weight
        self.prompts[target] = self.prompts[target] * ema_weight + self.prompts[source] * (1 - ema_weight)
        self.prompts[target].requires_grad = False
        
    def forward(self, query: torch.Tensor, **kwargs) -> tuple[torch.Tensor, List[torch.Tensor]]:
        B, D = query.shape
        assert D == self.feature_dim, f"Query dimention {D} does not match feature dimention {self.feature_dim}"

        if self.use_rp:
            score = query @ self.random_projection
        else:
            score = query
        score = self.non_linear_func(score)
        if self.gating_func == 'wta':
            topk_values, topk_indices = torch.topk(score, self.num_active, dim=-1)
            score = torch.zeros_like(score).scatter(1, topk_indices, topk_values)
        # selection_score = score @ self.map_to_expert
        # selection_indices = torch.topk(selection_score, self.selection_size).indices

        match = 1 - F.cosine_similarity(score.unsqueeze(1), self.key, dim=-1)
        selection_indices = match.topk(self.selection_size, dim=-1, largest=False).indices
        similarity = match.gather(1, selection_indices)

        selected_prompts = []
        for i in range(self.pool_num):
            selected_prompt = self.prompts[i][selection_indices]  # [B, selection_size, prompt_len, feature_dim]
            selected_prompt = selected_prompt.view(B, self.selection_size * self.prompt_len, self.feature_dim)
            selected_prompts.append(selected_prompt)
        # selected_prompts = torch.cat(selected_prompts, dim=1)
        return similarity, selected_prompts

class FlyPrompt(nn.Module):
    def __init__(self,
                 pool_num       : int   = 2,
                 len_e_prompt   : int   = 5,
                 e_pool         : int   = 30,
                 selection_size : int   = 5,
                 expansion_dim  : int   = 0,
                 gating_func    : str   = 'relu',
                 sparsity       : float = 0.05,
                 ema_batch      : float = 0.99,
                 ema_task       : float = 0.2,
                 task_num       : int   = 10,
                 lambd          : float = 0.5,
                 num_classes    : int   = 100,
                 backbone_name  : str   = None,
                 **kwargs):

        super().__init__()

        self.lambd          = lambd
        self.kwargs         = kwargs
        self.pool_size      = e_pool
        self.task_num       = task_num
        self.pool_num       = pool_num
        self.sparsity       = sparsity
        self.ema_task       = ema_task
        self.ema_batch      = ema_batch
        self.gating_func    = gating_func
        self.num_classes    = num_classes
        self.prompt_len     = len_e_prompt
        self.expansion_dim  = expansion_dim
        self.selection_size = selection_size

        self.task_count = 0
        
        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # Prompt
        assert e_pool > selection_size, 'e_pool must be larger than selection_size'

        self.prompt = EnsemblePrompt(
            pool_num,
            e_pool,
            len_e_prompt,
            selection_size,
            self.backbone.num_features,
            expansion_dim,
            gating_func,
            sparsity,
            ema_batch,
        )

        self.register_buffer('similarity', torch.zeros(1), persistent=False)

        logger.info(f"FlyPrompt initialized with parameters: {self.__dict__}")

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        self.backbone.eval()
        x = self.backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        with torch.no_grad():
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0].clone()
        similarity, prompts = self.prompt(query)
        self.similarity = similarity.mean()

        len_active_prompts = 1 if self.training else len(prompts)
        output_list = []
        for i in range(len_active_prompts):
            prompt_i = prompts[i].contiguous().view(B, self.selection_size * self.prompt_len, D)
            prompt_i = prompt_i + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
            x_i = torch.cat((x[:,0].unsqueeze(1), prompt_i, x[:,1:]), dim=1)
            x_i = self.backbone.blocks(x_i)
            x_i = self.backbone.norm(x_i)
            x_i = x_i[:, 1:self.selection_size * self.prompt_len + 1].clone()
            x_i = x_i.mean(dim=1)
            x_i = self.backbone.fc_norm(x_i)
            x_i = self.backbone.fc(x_i)
            output_list.append(x_i)

        if self.training:
            return output_list[0]
        else:
            # TODO: better pool-wise ensembling
            output_list = [output.softmax(dim=-1) for output in output_list]
            # output = torch.stack(output_list, dim=-1).mean(dim=-1)
            output = torch.stack(output_list, dim=-1).max(dim=-1)[0]
            return output

    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target) + self.lambd * self.similarity
    
    def batch_update_prompts(self):
        if self.pool_num > 1:
            self.prompt.ema_update_prompts(source=0, target=1, ema_weight=self.ema_batch)

    def task_update_prompts(self):
        if self.pool_num > 2:
            self.prompt.ema_update_prompts(source=1, target=2, ema_weight=self.ema_task)

    def process_task_count(self):
        self.task_count += 1