import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from .vit import VisionTransformer, Block, checkpoint_filter_fn
from functools import partial

def _create_custom_vit(model_class, variant, pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        model_class, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model


# https://github.com/JH-LEE-KR/dualprompt-pytorch/blob/master/attention.py 
class PreAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # prefix key, value
            prompt_length = prompt.size(2)
            prompt = prompt.view(B, 2, prompt_length, self.num_heads, C//self.num_heads) # (B, 2, 10, 12, 64)
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            expected_shape = (B, self.num_heads, C // self.num_heads)
            
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SinglePromptViT(VisionTransformer):
    def __init__(self, **kwargs):
        self.logit_type = kwargs.pop("logit_type")
        self.prompt_length = kwargs.pop("prompt_length")
        pos_prompt = kwargs.pop("pos_prompt")
        super().__init__(**kwargs)

        self.num_heads = kwargs.get("num_heads", 12)
        self.qkv_bias = kwargs.get("qkv_bias", True)
        self.init_values = kwargs.get("init_values", None)
        self.drop_rate = kwargs.get("drop_rate", 0.)
        self.attn_drop_rate = kwargs.get("attn_drop_rate", 0.)
        self.norm_layer = kwargs.get("norm_layer", partial(nn.LayerNorm, eps=1e-6))
        self.act_layer = kwargs.get("act_layer", nn.GELU)
        self.depth = kwargs.get("depth", 12)

        attn_layer=PreAttention
        dpr = [x.item() for x in torch.linspace(0, kwargs.get('drop_path_rate', 0.0), self.depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, init_values=self.init_values,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer, act_layer=self.act_layer, attn_layer=attn_layer)
            for i in range(self.depth)])
        
        for param in self.parameters():
            param.requires_grad = False
        
        num_classes = kwargs.get("num_classes")
        
        assert self.logit_type in ['linear', 'cos_sim']
        if self.logit_type == 'linear':
            self.fc = nn.Linear(self.embed_dim, num_classes)
            self.fc.weight.requires_grad = True
            self.fc.bias.requires_grad = True
        else:
            self.fc = nn.Linear(self.embed_dim, num_classes, bias=False)
            self.fc.weight.requires_grad = True
        
        assert self.prompt_length != 0 and len(pos_prompt) != 0
        self.register_buffer('pos_prompt', torch.tensor(pos_prompt, dtype=torch.int64))
        self.prompt = nn.Parameter(torch.randn(len(self.pos_prompt) * 2 * self.prompt_length, self.embed_dim, requires_grad= True))
        nn.init.uniform_(self.prompt, -1, 1)
        print('Prompt size: ', self.prompt.size())
        print()


    def forward_features(self, inputs: torch.Tensor):
        x = self.patch_embed(inputs)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # prefix tuning
        B, N, C = x.size()
        prompt = self.prompt.expand(B, -1, -1) # (B, 5*2*10, 768)
        prompt = prompt.contiguous().view(B, len(self.pos_prompt), 2, self.prompt_length, C)
        prompt = prompt + self.pos_embed[:,:1,:].unsqueeze(1).unsqueeze(1).expand(B, len(self.pos_prompt), 2, self.prompt_length, C) # (B, 5, 2, 10, 768)
        for n, block in enumerate(self.blocks):
            if n in self.pos_prompt:
                curr_prompt = prompt[:, n] # (B, 2, 10, 768)
                x = block(x, curr_prompt)
            else:
                x = block(x)
        
        x = self.norm(x) 
        # x = x.mean(dim=1) # avg(cls + patch)
        x = x[:, 0]
        
        return x
        
    def forward_head(self, x: torch.Tensor):
        
        if self.logit_type=='linear':
            return self.fc(x)
        elif self.logit_type=='cos_sim':
            return F.normalize(x, dim=1) @ F.normalize(self.fc.weight, dim=1).T / 0.1
        else:
            raise ValueError(f'Invalid logit type : {self.logit_type}')
    
    def forward(self, inputs : torch.Tensor):
        x = self.forward_features(inputs)
        x = self.forward_head(x)
        return x

@register_model
def singlePrompt_vit(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
    model = _create_custom_vit(SinglePromptViT, 'vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
