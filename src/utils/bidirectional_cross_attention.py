import torch
from torch import nn
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64,
        context_dim=None,
        dropout=0.0,
        talking_heads=False,
        prenorm=False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv1d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv1d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask=None,
        context_mask=None,
        return_attn=False,
        rel_pos_bias=None
    ):
        # print("x.shape: ", x.shape)
        b = x.shape[0]  # Batch size
        i, j = x.shape[1], context.shape[1]  # Sequence lengths
        h = self.heads
        device = x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # Get shared query/keys and values for sequence and context
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # Split out head
        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (qk, context_qk, v, context_v))

        # Get similarities
        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        # Relative positional bias, if supplied
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # Mask
        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device=device, dtype=torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device=device, dtype=torch.bool))

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # Get attention along both sequence length and context length dimensions
        attn = sim.softmax(dim=-1)
        context_attn = sim.softmax(dim=-2)

        # Dropouts
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # Talking heads
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # Src sequence aggregates values from context, context aggregates values from src sequence
        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        # Merge heads and combine out
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = torch.squeeze(self.to_out(out))
        context_out = torch.squeeze(self.context_to_out(context_out))

        if return_attn:
            return out, context_out, attn, context_attn

        # print("out.shape: ", out.shape)
        # print("context_out.shape: ", context_out.shape)
        return out, context_out