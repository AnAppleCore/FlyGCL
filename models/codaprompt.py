# test
from typing import TypeVar, Iterable
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs
from collections import OrderedDict
from models.vit import _create_vision_transformer
import copy


logger = logging.getLogger()

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

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

load = True
G_dist = False
e_proj= False
g_proj = False
e_ratio = False
g_ratio = False
learnable_mask = False
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

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    

class CodaPrompt(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (),
                 len_g_prompt   : int   = 0,
                 pos_e_prompt   : Iterable[int] = (0,1,2,3,4),
                 len_e_prompt   : int   = 20,
                 prompt_func    : str   = 'prompt_tuning',
                 task_num       : int   = 10,
                 class_num      : int   = 100,
                 lambd          : float = 1.0,
                 backbone_name  : str   = None,
                 key_dim                = 768,
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

        # self.features = torch.empty(0)
        # self.keys     = torch.empty(0)

        if backbone_name is None:
            raise ValueError('backbone_name must be specified')


        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.ones(1).view(1))
        # self.register_buffer('mask', torch.zeros(class_num))
        self.mask = 0
        
        self.lambd      = lambd
        self.class_num  = class_num
        self.key_d = key_dim
        self.ortho_mu = 0
        self.task_count = 0

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
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

        # Slice the eprompt
        self.e_pool = e_pool
        self.num_pt_per_task = int(e_pool / task_num)
        self.task_num = task_num
        self.task_id = 0 # if _convert_train_task is not called, task will undefined
        
        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_prompt = None 
            for e in self.pos_e_prompt:
                p = tensor_prompt(e_pool, self.len_e_prompt, self.backbone.num_features)
                k = tensor_prompt(e_pool, self.key_d)
                a = tensor_prompt(e_pool, self.key_d)
                p = self.gram_schmidt(p)
                k = self.gram_schmidt(k)
                a = self.gram_schmidt(a)
                setattr(self, f'e_p_{e}',p)
                setattr(self, f'e_k_{e}',k)
                setattr(self, f'e_a_{e}',a)

        else: 
            raise ValueError('Unknown prompt_func: {}'.format(prompt_func))

        if self.load_pt:
            for e in self.pos_e_prompt:
                k = getattr(self,f'e_k_{e}')
                a = getattr(self,f'e_a_{e}')
                p = getattr(self,f'e_p_{e}')
                # Load ISA prompt
                load_path = '/scratch/algorab/zkang/MVP/prompt/CODA-N101M0_T1_prompt_IMGNT100_samcl_add_random_allprompt_proj_moreaug_epoch3_rho.1/T0_{}_{}_prompt.pt'
                k = nn.Parameter(torch.load(load_path.format('K', e)).detach().clone(), requires_grad= True)
                a = nn.Parameter(torch.load(load_path.format('A', e)).detach().clone(), requires_grad= True)
                load_path = '/scratch/algorab/zkang/MVP/prompt/CODA-N101M0_T1_prompt_IMGNT100_samcl_add_random_allprompt_proj_moreaug_epoch3_rho.1/T0_{}_proj_{}_prompt.pt'
                p = nn.Parameter(torch.load(load_path.format('p', e)).detach().clone(), requires_grad= True)
                
                # p = self.gram_schmidt(p)
                # k = self.gram_schmidt(k)
                # a = self.gram_schmidt(a)
                setattr(self, f'e_p_{e}',p)
                setattr(self, f'e_k_{e}',k)
                setattr(self, f'e_a_{e}',a)
            print('load from {}...'.format(load_path))
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
        
        e_prompt = e_prompt.contiguous().view(B, self.e_length, self.len_e_prompt, C)
        e_prompt = e_prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).expand(B, self.e_length, self.len_e_prompt, C)
            

        for n, block in enumerate(self.backbone.blocks):
            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e]), dim = 1)

            x = block(x)
            x = x[:, :N, :]
        return x
    
 

    def forward(self, inputs : torch.Tensor) :
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

        g_p = None
        e_p = None
        s = self.task_count * self.num_pt_per_task
        f = (self.task_count+1) * self.num_pt_per_task
        loss = 0
        for e in self.pos_e_prompt:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            p = getattr(self,f'e_p_{e}')
            if self.training:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', query, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p) # B, len_e_prompt, d
            
            if self.e_proj :
                P_ = self.proj_e_pt(P_) + P_
            
            if e_p is None:
                e_p = P_
            else:
                e_p = torch.cat((e_p, P_), dim=1)
            
            if self.training and self.ortho_mu > 0:
                loss += ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu

            # if self.training and start_id < self.e_pool:
            #     if self.memory_size > 0  and self.load_pt:
            #         res_e = self.e_prompt(query)
            #     elif self.ISA:
            #         res_e = self.e_prompt(query)
            #     else:
            #         res_e = self.e_prompt(query, s=start_id, e=end_id)

            # # elif not self.training and start_id < self.e_pool:
            # #     res_e = self.e_prompt(query, s=0, e=end_id)
            # else:
            #     res_e = self.e_prompt(query)

            # if self.learnable_mask:
            #     e_s, e_p, learned_mask = res_e
            # else:
            #     e_s, e_p = res_e
            # if self.e_proj  and e_ratio:
            #     e_p = self.e_ratio*self.proj_e_pt(e_p)+e_p
            # elif self.e_proj :
            #     e_p = self.proj_e_pt(e_p)+e_p

        e_p = e_p.unsqueeze(1)
        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_p, e_p)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]
        x = self.backbone.fc(cls_token)

        if self.training:
            return x, loss
        else:
            return x
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.pos_e_prompt:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    
    def get_count(self):
        return self.e_prompt.update()

    def loss_fn(self, output, target):

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


    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = self.num_pt_per_task
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 