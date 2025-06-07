# test
from typing import TypeVar, Iterable
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.tensorboard import SummaryWriter
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs
from collections import OrderedDict
from models.vit import _create_vision_transformer

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

T = TypeVar('T', bound = 'nn.Module')

default_cfgs['vit_base_patch16_224_l2p'] = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        num_classes=21843)

# Register the backbone model to timm
@register_model
def vit_base_patch16_224_l2p(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_l2p', pretrained=pretrained, **model_kwargs)
    
    return model

G_dist = False
e_ratio = False
g_ratio = False

L1Loss = torch.nn.L1Loss()
MSELoss = torch.nn.MSELoss()

class Prompt(nn.Module):
    def __init__(self,
                 pool_size            : int,
                 selection_size       : int,
                 prompt_len           : int,
                 dimention            : int,
                 _diversed_selection  : bool = False,
                 _batchwise_selection : bool = False,
                 kwargs=None):
        super().__init__()
        self.learnable_mask = kwargs.get("learnable_mask")
        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention
        self._diversed_selection  = _diversed_selection
        self._batchwise_selection = _batchwise_selection

        self.key     = nn.Parameter(torch.randn(pool_size, dimention, requires_grad= True))
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad= True))
        
        torch.nn.init.uniform_(self.key,     -1, 1)
        torch.nn.init.uniform_(self.prompts, -1, 1)

        self.register_buffer('frequency', torch.ones (pool_size))
        self.register_buffer('counter',   torch.zeros(pool_size))
        if self.learnable_mask:
            self.mask    = nn.Parameter(torch.zeros(pool_size, 200) - 1)
    
    def forward(self, query : torch.Tensor, s=None, e=None, **kwargs):
        B, D = query.shape
        assert D == self.dimention, f'Query dimention {D} does not match prompt dimention {self.dimention}'
        # Select prompts
        if s is None and e is None:
            match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        else:
            assert s is not None
            assert e is not None
            match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key[s:e], dim=-1)
        # match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        if self.training and self._diversed_selection:
            topk = match * F.normalize(self.frequency, p=1, dim=-1)
        else:
            topk = match
        _ ,topk = topk.topk(self.selection_size, dim=-1, largest=False, sorted=True)
        # Batch-wise prompt selection
        if self._batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _,  mosts  = counts.topk(self.selection_size, largest=True, sorted=True)
            topk = idx[mosts].clone().expand(B, -1)
        # Frequency counter
        self.counter += torch.bincount(topk.reshape(-1).clone(), minlength = self.pool_size)
        # selected prompts
        selection = self.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, self.dimention).clone())
        simmilarity = match.gather(1, topk)
        # get unsimilar prompts also 
        
        if self.learnable_mask:
            # Get learnable mask:
            mask = self.mask[topk].mean(1).squeeze().clone()
            mask = torch.sigmoid(mask)*2.
            return simmilarity, selection, mask 
        return simmilarity, selection

    def update(self):
        if self.training:
            self.frequency += self.counter
        counter = self.counter.clone()
        self.counter *= 0
        if self.training:
            return self.frequency - 1
        else:
            return counter

    def load_e(self, path):
        e_pt = torch.load(path)#.state_dict()
        pt_prompt = e_pt[0] 
        pt_prompt_key = e_pt[1]     
        # self.key = nn.Parameter(pt_prompt_key.detach().clone().view((self.pool_size, self.dimention)))
        num_layers, dual, poolsize, len_prompt,  num_heads, heads_embed_dim = pt_prompt.shape
        pt_prompt = pt_prompt.detach().clone().view((num_layers, dual, poolsize, len_prompt, num_heads*heads_embed_dim)).permute((2, 1, 0, 3, 4))
        pt_prompt = pt_prompt.reshape((self.pool_size, self.prompt_len, self.dimention))
        self.prompts = nn.Parameter(pt_prompt)
        # self.key = nn.Parameter(pt_prompt_key.detach().clone().view((self.pool_size, self.dimention)))

    def load_g(self, path):
        g_pt = torch.load(path)  
        self.prompts = nn.Parameter(g_pt.detach().clone().view((self.pool_size, self.prompt_len, self.dimention)))
    
    def load(self, path, include_key=True):
        pt = torch.load(path) 
        self.prompts = nn.Parameter(pt.prompts.detach().clone())
        if include_key:
            self.key = nn.Parameter(pt.key.detach().clone())
    
    def load_prompts_only(self, path):
        pt = torch.load(path) 
        self.prompts = nn.Parameter(pt.detach().clone())

    def load_from_ckpt(self, ckpt, include_key=True):
        self.prompts = nn.Parameter(ckpt.prompts.detach().clone())
        if include_key:
            self.key = nn.Parameter(ckpt.key.detach().clone())

class PromptAdd(nn.Module):
    def __init__(self,
                 pool_size            : int,
                 selection_size       : int,
                 prompt_len           : int,
                 dimention            : int,
                 _diversed_selection  : bool = False,
                 _batchwise_selection : bool = False,
                 kwargs=None):
        super().__init__()
        self.learnable_mask = kwargs.get("learnable_mask")
        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention
        self._diversed_selection  = _diversed_selection
        self._batchwise_selection = _batchwise_selection

        self.key     = nn.Parameter(torch.randn(pool_size, dimention, requires_grad= True))
        self.prompts = nn.Parameter(torch.ones(pool_size, 2*prompt_len, dimention, requires_grad= True))
        self.ff = nn.Parameter(torch.ones(pool_size, prompt_len, dimention, requires_grad= True))
        
        torch.nn.init.uniform_(self.key,     -1, 1)

        self.register_buffer('frequency', torch.ones (pool_size))
        self.register_buffer('counter',   torch.zeros(pool_size))
        if self.learnable_mask:
            self.mask    = nn.Parameter(torch.zeros(pool_size, 200) - 1)
    
    def forward(self, query : torch.Tensor, s=None, e=None, **kwargs):
        B, D = query.shape
        assert D == self.dimention, f'Query dimention {D} does not match prompt dimention {self.dimention}'
        # Select prompts
        if s is None and e is None:
            match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        else:
            assert s is not None
            assert e is not None
            match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key[s:e], dim=-1)
        # match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        if self.training and self._diversed_selection:
            topk = match * F.normalize(self.frequency, p=1, dim=-1)
        else:
            topk = match
        _ ,topk = topk.topk(self.selection_size, dim=-1, largest=False, sorted=True)
        # Batch-wise prompt selection
        if self._batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _,  mosts  = counts.topk(self.selection_size, largest=True, sorted=True)
            topk = idx[mosts].clone().expand(B, -1)
        # Frequency counter
        self.counter += torch.bincount(topk.reshape(-1).clone(), minlength = self.pool_size)
        # selected prompts
        selection = self.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len*2, self.dimention).clone())
        selection_ff = self.ff.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, self.dimention).clone())
        simmilarity = match.gather(1, topk)
        # get unsimilar prompts also 
        
        if self.learnable_mask:
            # Get learnable mask:
            mask = self.mask[topk].mean(1).squeeze().clone()
            mask = torch.sigmoid(mask)*2.
            return simmilarity, selection, mask 
        return simmilarity, [selection, selection_ff]

    def update(self):
        if self.training:
            self.frequency += self.counter
        counter = self.counter.clone()
        self.counter *= 0
        if self.training:
            return self.frequency - 1
        else:
            return counter


class DualPrompt(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (0, 1),
                 len_g_prompt   : int   = 10,
                 pos_e_prompt   : Iterable[int] = (2,3,4),
                 len_e_prompt   : int   = 20,
                 prompt_func    : str   = 'prompt_tuning',
                 task_num       : int   = 10,
                 class_num      : int   = 100,
                 lambd          : float = 1.0,
                 backbone_name  : str   = None,
                 **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.load_pt = kwargs.get("load_pt")
        self.learnable_mask = kwargs.get("learnable_mask")
        self.imbalance = kwargs.get("imbalance")
        self.memory_size = kwargs.get("memory_size")
        self.ISA = kwargs.get("isa")
        self.e_proj = kwargs.get("e_proj")
        self.g_proj = kwargs.get("g_proj")

        if self.imbalance:            
            self.weight_lst = torch.zeros(200).cuda()

        # self.features = torch.empty(0)
        # self.keys     = torch.empty(0)

        if backbone_name is None:
            raise ValueError('backbone_name must be specified')

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.ones(1).view(1))
        # self.register_buffer('mask', torch.zeros(class_num))
        self.mask = 0
        
        self.lambd      = lambd
        self.class_num  = class_num

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        # self.add_module('backbone', timm.create_model('vit_base_patch16_clip_224', pretrained=True, num_classes=class_num))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        self.tasks = []

        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt
        g_pool = 1
        e_pool = 10
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0
        
        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(g_pool, 1, self.g_length * self.len_g_prompt, self.backbone.num_features, _batchwise_selection = False, kwargs=self.kwargs)
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(e_pool, 1, self.e_length * self.len_e_prompt, self.backbone.num_features, _batchwise_selection = False, kwargs=self.kwargs)

        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(g_pool, 1, 2 * self.g_length * self.len_g_prompt, self.backbone.num_features, _batchwise_selection = False, kwargs=self.kwargs)
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(e_pool, 1, 2 * self.e_length * self.len_e_prompt, self.backbone.num_features, _batchwise_selection = False, kwargs=self.kwargs)

        elif prompt_func == 'prefix_tuning_add':
            self.prompt_func = self.prefix_tuning_add
            # self.e_prompt = None if len(pos_e_prompt) == 0 else PromptAdd(5, 1,  12, self.backbone.num_features, _batchwise_selection = True, kwargs=self.kwargs)
            self.e_prompt = None if len(pos_e_prompt) == 0 else nn.Parameter(torch.ones(12, 1, 2, self.backbone.num_features, requires_grad= True))
            self.e_ff = None if len(pos_e_prompt) == 0 else nn.Parameter(torch.ones(12, 1, self.backbone.num_features, requires_grad= True))
            self.pos_e_prompt = torch.tensor([i for i in range(12)], dtype=torch.int64)
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(g_pool, 1, 2 * self.g_length * self.len_g_prompt, self.backbone.num_features, _batchwise_selection = False, kwargs=self.kwargs)


        else: raise ValueError('Unknown prompt_func: {}'.format(prompt_func))
        self.prompt_func_type = prompt_func
        self.g_prompt.key = None

        # Slice the eprompt
        self.e_pool = e_pool
        self.num_pt_per_task = int(e_pool / task_num)
        self.task_num = task_num
        self.task_id = 0 # if _convert_train_task is not called, task will undefined

        if self.load_pt:
                        
            e_load_path = 'pretrained_prompt/e_prompt.pt'
            g_load_path = 'pretrained_prompt/g_prompt.pt'
                        
            print('loading from: {}'.format(g_load_path))
            # self.g_prompt.load(g_load_path, include_key=False)
            self.g_prompt.load_prompts_only(g_load_path)
            self.g_prompt.key = None
            print('loading from: {}'.format(e_load_path))
            # self.e_prompt.load(e_load_path)
            self.e_prompt.load_prompts_only(e_load_path)
            
            if G_dist:
                self.g_pt = torch.load(g_load_path) 
                if isinstance(self.g_pt, Prompt):
                    self.g_pt_prompt = self.g_pt.prompts.detach().clone()
                else:
                    self.g_pt_prompt = self.g_pt.detach().clone()
                self.e_pt = torch.load(e_load_path) 

            self.scale = nn.Parameter(torch.ones(self.g_length))
            self.translation = nn.Parameter(torch.zeros(self.g_length * self.len_g_prompt, self.backbone.num_features))

        self.proj_g_pt = None
        if self.g_proj:
            factor = 8
            self.proj_g_pt = torch.nn.Sequential(OrderedDict([
                                ('fc1', torch.nn.Linear(self.backbone.num_features, int(self.backbone.num_features/factor))),
                                # ('dropout', torch.nn.Dropout(p=0.5)),
                                ('ln1', torch.nn.LayerNorm(int(self.backbone.num_features/factor))),
                                ('relu1', torch.nn.ReLU()),
                                ('fc2', torch.nn.Linear(int(self.backbone.num_features/factor), self.backbone.num_features)),
                            ]))
            self.g_ratio = nn.Parameter(torch.ones(1))
        
        self.proj_e_pt = None
        if self.e_proj :
            factor = 8
            self.proj_e_pt = torch.nn.Sequential(OrderedDict([
                                ('fc1', torch.nn.Linear(self.backbone.num_features, int(self.backbone.num_features/factor))),
                                ('ln1', torch.nn.LayerNorm(int(self.backbone.num_features/factor))),
                                ('relu1', torch.nn.ReLU()),
                                ('fc2', torch.nn.Linear(int(self.backbone.num_features/factor), self.backbone.num_features)),
                            ]))
            self.e_ratio = nn.Parameter(torch.ones(1))

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


        if G_dist and self.load_pt:
            g_prompt_pt = self.g_pt_prompt[0]
            g_prompt_pt = g_prompt_pt.expand(B, -1, -1)
            g_prompt_pt = g_prompt_pt.contiguous().view(B, self.g_length, self.len_g_prompt, C)
            g_prompt_pt = g_prompt_pt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.g_length, self.len_g_prompt, C).detach().clone()
            
        dist_loss = 0
        with torch.no_grad():
            x_gpt = x
        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                selected_g_prompt = g_prompt[:, pos_g]
                # trans = self.translation.expand(B, -1, -1).contiguous().view(B, self.g_length, self.len_g_prompt, C)[:, pos_g]
                # assert trans.shape == selected_g_prompt.shape
                # selected_g_prompt = selected_g_prompt*self.scale[pos_g] + trans
                x = torch.cat((x, selected_g_prompt), dim = 1)

                if G_dist and self.load_pt:
                    selected_g_prompt_pt = g_prompt_pt[:, pos_g]
                    # selected_g_prompt_pt = selected_g_prompt_pt*self.scale[pos_g]  + trans
                    x_gpt = torch.cat((x_gpt, selected_g_prompt_pt), dim = 1)


            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e]), dim = 1)

            x = block(x)
            x = x[:, :N, :]

            if pos_g.numel() != 0 and G_dist and self.load_pt:
                with torch.no_grad():
                    x_gpt = block(x_gpt)
                    x_gpt = x_gpt[:, :N, :]
                    # x_gpt = self.backbone.norm(x_gpt)
                # if pos_g == self.pos_g_prompt[-1]:
                dist_loss += L1Loss(x_gpt, x) # self.backbone.norm(x))

        if G_dist and self.load_pt:
            return x, dist_loss
        else:
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
                xk = torch.cat((g_prompt[:, pos_g * 2 + 0], xk), dim = 1)
                xv = torch.cat((g_prompt[:, pos_g * 2 + 1], xv), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat((e_prompt[:, pos_e * 2 + 0], xk), dim = 1)
                xv = torch.cat((e_prompt[:, pos_e * 2 + 1], xv), dim = 1)
            
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

    def prefix_tuning_add(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        # g_prompt = g_prompt.contiguous().view(B, 2 * self.g_length, self.len_g_prompt, C)
        # e_prompt = e_prompt.contiguous().view(B, 2 * self.e_length, self.len_e_prompt, C)
        
        # # Key match
        # kv, ff = e_prompt
        # kv = kv.contiguous().view(B, 12, 2 , C)
        # ff = ff.contiguous().view(B, 12, C)

        

        for n, block in enumerate(self.backbone.blocks):
            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            # # Key match
            # e_p = kv[:, n][0, :].unsqueeze(0)
            # ff_p = ff[:, n][0, :].unsqueeze(0)
            
            # Unique
            e_p = self.e_prompt[n,:]
            ff_p = self.e_ff[n, :]
           
            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias
            
            
            B, N, C = xq.shape

            xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:]*e_p[:, 0], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            # xk = torch.bmm(xk, e_p[:, 0].unsqueeze(1) * weight[C:2*C,:].unsqueeze(0)).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:]*e_p[:, 1], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            # xv = torch.bmm(xv, e_p[:, 1].unsqueeze(1) * weight[2*C: ,:].unsqueeze(0)).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path1(block.ls1(attention))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x*ff_p))))

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
        # if self.training:
        #     self.features = torch.cat((self.features, query.detach().cpu()), dim = 0)

        if self.g_prompt is not None:
            g_p = self.g_prompt.prompts[0]
            g_p = g_p.expand(B, -1, -1)

            if self.g_proj and g_ratio:
                g_p = self.g_ratio*self.proj_g_pt(g_p)+g_p
            elif self.g_proj:
                g_p = self.proj_g_pt(g_p)+g_p
            
        else:
            g_p = None
        if self.e_prompt is not None and self.prompt_func_type != 'prefix_tuning_add':
                start_id = self.task_id * self.num_pt_per_task
                end_id = (self.task_id+1) * self.num_pt_per_task
                if self.training and start_id < self.e_pool:
                    if self.memory_size > 0  and self.load_pt:
                        res_e = self.e_prompt(query)
                    elif self.ISA:
                        res_e = self.e_prompt(query)
                    else:
                        res_e = self.e_prompt(query, s=start_id, e=end_id)

                # elif not self.training and start_id < self.e_pool:
                #     res_e = self.e_prompt(query, s=0, e=end_id)
                else:
                    res_e = self.e_prompt(query)

                if self.learnable_mask:
                    e_s, e_p, learned_mask = res_e
                else:
                    e_s, e_p = res_e
                if self.e_proj  and e_ratio:
                    e_p = self.e_ratio*self.proj_e_pt(e_p)+e_p
                elif self.e_proj :
                    e_p = self.proj_e_pt(e_p)+e_p

        else:
            e_p = None
            e_s = 0

        if G_dist and self.load_pt:
            x, dist_loss = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
            x = self.backbone.norm(x)
            x = self.backbone.fc(x[:, 0])

            self.similarity = e_s.mean()
            dist_loss = L1Loss(self.g_prompt.prompts[0], self.g_pt_prompt[0].detach().clone())
            return x, dist_loss

        else:
            x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
            x = self.backbone.norm(x)
            cls_token = x[:, 0]
            x = self.backbone.fc(x[:, 0])
            if self.learnable_mask:
                x = x * learned_mask
            if self.prompt_func_type != 'prefix_tuning_add':
                self.similarity = e_s.mean()
            
            if return_feat:
                return x, cls_token
            else:
                return x

    def convert_train_task(self, task : torch.Tensor, **kwargs):
    
        task = torch.tensor(task,dtype=torch.float)
        flag = -1
        for n, t in enumerate(self.tasks):
            if torch.equal(t, task):
                flag = n
                break
        if flag == -1:
            self.tasks.append(task)
            self.task_id = len(self.tasks) - 1
            if self.training:
                if self.task_id != 0:
                    with torch.no_grad():
                        # self.e_prompt.prompts[self.task_id] = self.e_prompt.prompts[self.task_id - 1].detach().clone()
                        # self.e_prompt.key[self.task_id] = self.e_prompt.key[self.task_id - 1].detach().clone()
                        self.e_prompt.prompts[self.task_id] = self.e_prompt.prompts[self.task_id - 1].clone()
                        self.e_prompt.key[self.task_id] = self.e_prompt.key[self.task_id - 1].clone()
        else :
            self.task_id = flag

        # self.mask += -torch.inf
        # self.mask[task] = 0
        return
        
    def get_count(self):
        return self.e_prompt.update()

    def loss_fn(self, output, target):
        
        if self.imbalance:
            cls_lst, cls_count = target.unique(return_counts=True)
            # print(cls_lst, cls_count)
            for cc, count in zip(cls_lst,cls_count):
               self.weight_lst[cc.item()] += count.item()
            # print(weight_lst)
            weights = 1/self.weight_lst
            weights[self.weight_lst==0] = 0
            return F.cross_entropy(output, target, weight=weights) + self.lambd * self.similarity

        return F.cross_entropy(output, target) + self.lambd * self.similarity
    
    def reload_pt(self):
        self.g_prompt.load_from_ckpt(self.g_pt)
        self.e_prompt.load_from_ckpt(self.e_pt)

    def freeze_ss(self):
        self.scale.requires_grad = False
        self.translation.requires_grad = False

    def freeze_prompt(self):
        for p in self.e_prompt.parameters():
            p.requires_grad = False
        for p in self.g_prompt.parameters():
            p.requires_grad = False

    def freeze_eg_proj(self):
        print('freezing e&g projector ')
        if self.proj_e_pt is not None:
            for p in self.proj_e_pt.parameters():
                p.requires_grad = False
        if self.proj_g_pt is not None:
            for p in self.proj_g_pt.parameters():
                p.requires_grad = False


