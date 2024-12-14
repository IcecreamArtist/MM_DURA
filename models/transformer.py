import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

def posemb_sincos_1d(patches, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TransformerWithClsToken(nn.Module):
	def __init__(self, dim, heads, depth):
		super().__init__()
		self.transformer = Transformer(
			dim=dim, depth=depth, heads=heads, dim_head=64, mlp_dim=2048
		)
		self.cls_token = nn.Parameter(torch.rand(1,1,dim))
	def forward(self, x):
		B, _, D = x.shape # [B,9,D]
		cls_tokens = self.cls_token.expand(B, -1, -1) # [B,1,D]
		x = torch.cat((cls_tokens, x), dim=1) # [B,10,D]
		pe = posemb_sincos_1d(x)
		x = rearrange(x, 'b ... d -> b (...) d') + pe
		x = self.transformer(x)
		x = x[:,0,:]
		return x


class DualCrossModalTransformer(nn.Module):
    def __init__(self, emb_dim, num_heads, 
                 dim_feedforward, dropout):
        super(DualCrossModalTransformer, self).__init__()
        self.first_level_A_in_B = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.first_level_B_in_A = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True)
        # concat dual direction attention
        self.second_level_A_in_B = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.second_level_B_in_A = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True)
        # linear map to keep emb dim
        self.fc1 = nn.Sequential(*[nn.Linear(emb_dim*2, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU()])
        self.fc2 = nn.Sequential(*[nn.Linear(emb_dim*2, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU()])
        
    def forward(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        pass
    
    def CatDualAtten_imgcli_gen(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_img_in_cli, _ = self.first_level_A_in_B(
             query=img_emb, key=cli_emb, value=cli_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_cli_in_img, _ = self.first_level_B_in_A(
             query=cli_emb, key=img_emb, value=img_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        concat_atten = torch.concat((attn_img_in_cli,attn_cli_in_img),dim=2)
        concat_atten_1 = self.fc1(concat_atten)
        attn_gen_in_fusion, _ = self.second_level_A_in_B(
             query=gen_emb, key=concat_atten_1, value=concat_atten_1, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_fusion_in_gen, _ = self.second_level_B_in_A(
             query=concat_atten_1, key=gen_emb, value=gen_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        # attn_fusion_in_gen_1 = repeat(attn_fusion_in_gen, 'b k d -> b n d', n=attn_gen_in_fusion.shape[1])
        two_level_fusion = torch.concat((attn_gen_in_fusion,attn_fusion_in_gen),dim=2)
        two_level_fusion_1 = self.fc2(two_level_fusion)
        return two_level_fusion_1
    
    def CatDualAtten_imggen_cli(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_img_in_gen, _ = self.first_level_A_in_B(
             query=img_emb, key=gen_emb, value=gen_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_gen_in_img, _ = self.first_level_B_in_A(
             query=gen_emb, key=img_emb, value=img_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        concat_atten = torch.concat((attn_img_in_gen,attn_gen_in_img),dim=2)
        concat_atten_1 = self.fc1(concat_atten)
        attn_cli_in_fusion, _ = self.second_level_A_in_B(
             query=cli_emb, key=concat_atten_1, value=concat_atten_1, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_fusion_in_cli, _ = self.second_level_B_in_A(
             query=concat_atten_1, key=cli_emb, value=cli_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        two_level_fusion = torch.concat((attn_cli_in_fusion,attn_fusion_in_cli),dim=2)
        two_level_fusion_1 = self.fc2(two_level_fusion)
        return two_level_fusion_1
    
    def CatDualAtten_cligen_img(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_cli_in_gen, _ = self.first_level_A_in_B(
             query=cli_emb, key=gen_emb, value=gen_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_gen_in_cli, _ = self.first_level_B_in_A(
             query=gen_emb, key=cli_emb, value=cli_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        concat_atten = torch.concat((attn_cli_in_gen,attn_gen_in_cli),dim=2)
        concat_atten_1 = self.fc1(concat_atten)
        attn_img_in_fusion, _ = self.second_level_A_in_B(
             query=img_emb, key=concat_atten_1, value=concat_atten_1, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_fusion_in_img, _ = self.second_level_B_in_A(
             query=concat_atten_1, key=img_emb, value=img_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        # attn_fusion_in_gen_1 = repeat(attn_fusion_in_gen, 'b k d -> b n d', n=attn_gen_in_fusion.shape[1])
        two_level_fusion = torch.concat((attn_img_in_fusion,attn_fusion_in_img),dim=2)
        two_level_fusion_1 = self.fc2(two_level_fusion)
        return two_level_fusion_1
    

    def CatDualAtten_imgcli_gen_v2(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_img_in_cli, _ = self.first_level_A_in_B(
             query=img_emb, key=cli_emb, value=cli_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_cli_in_img, _ = self.first_level_B_in_A(
             query=cli_emb, key=img_emb, value=img_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        concat_atten = torch.concat((attn_img_in_cli,attn_cli_in_img),dim=2)
        concat_atten_1 = self.fc1(concat_atten)
        attn_gen_in_fusion, _ = self.second_level_A_in_B(
             query=gen_emb, key=concat_atten_1, value=concat_atten_1, 
             attn_mask=cli_mask, key_padding_mask=img_mask)

        return attn_gen_in_fusion
    

    def CatDualAtten_imggen_cli_v2(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_img_in_gen, _ = self.first_level_A_in_B(
             query=img_emb, key=gen_emb, value=gen_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_gen_in_img, _ = self.first_level_B_in_A(
             query=gen_emb, key=img_emb, value=img_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        concat_atten = torch.concat((attn_img_in_gen,attn_gen_in_img),dim=2)
        concat_atten_1 = self.fc1(concat_atten)
        attn_cli_in_fusion, _ = self.second_level_A_in_B(
             query=cli_emb, key=concat_atten_1, value=concat_atten_1, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        return attn_cli_in_fusion
    
    def CatDualAtten_cligen_img_v2(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_cli_in_gen, _ = self.first_level_A_in_B(
             query=cli_emb, key=gen_emb, value=gen_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_gen_in_cli, _ = self.first_level_B_in_A(
             query=gen_emb, key=cli_emb, value=cli_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        concat_atten = torch.concat((attn_cli_in_gen,attn_gen_in_cli),dim=2)
        concat_atten_1 = self.fc1(concat_atten)
        attn_img_in_fusion, _ = self.second_level_A_in_B(
             query=img_emb, key=concat_atten_1, value=concat_atten_1, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        return attn_img_in_fusion
    

class CrossModalTransformer(nn.Module):
    def __init__(self, emb_dim, num_heads, 
                 dim_feedforward, dropout):
        super(CrossModalTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_output, _ = self.cross_attention(
             query=img_emb, key=cli_emb, value=cli_emb, 
             attn_mask=cli_mask, key_padding_mask=img_mask)
        attn_output, _ = self.cross_attention2(
             query=gen_emb, key=attn_output, value=attn_output)
        return attn_output
    
    def fuse3(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        return self(img_emb, cli_emb, gen_emb, img_mask, cli_mask)
    
    def fuse2(self, emb1, emb2, mask1=None, mask2=None):
        attn_output, _ = self.cross_attention(
             query=emb1, key=emb2, value=emb2, 
             attn_mask=mask2, key_padding_mask=mask1)
        return attn_output
    
    def fuse_varyCross_cli_img_gen(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_output, _ = self.cross_attention(
             query=cli_emb, key=img_emb, value=img_emb, 
             attn_mask=img_mask, key_padding_mask=cli_mask)
        attn_output, _ = self.cross_attention2(
             query=gen_emb, key=attn_output, value=attn_output)
        return attn_output
    
    def fuse_varyCross_gen_img_cli(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_output, _ = self.cross_attention(
             query=gen_emb, key=img_emb, value=img_emb, 
             attn_mask=img_mask, key_padding_mask=cli_mask)
        attn_output, _ = self.cross_attention2(
             query=cli_emb, key=attn_output, value=attn_output)
        return attn_output
    
    def fuse_varyCross_img_gen_cli(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_output, _ = self.cross_attention(
             query=img_emb, key=gen_emb, value=gen_emb, 
             attn_mask=img_mask, key_padding_mask=cli_mask)
        attn_output, _ = self.cross_attention2(
             query=cli_emb, key=attn_output, value=attn_output)
        return attn_output
    
    def fuse_varyCross_cli_gen_img(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_output, _ = self.cross_attention(
             query=cli_emb, key=gen_emb, value=gen_emb, 
             attn_mask=img_mask, key_padding_mask=cli_mask)
        attn_output, _ = self.cross_attention2(
             query=img_emb, key=attn_output, value=attn_output)
        return attn_output
    
    def fuse_varyCross_gen_cli_img(self, img_emb, cli_emb, gen_emb, img_mask=None, cli_mask=None):
        # permute
        attn_output, _ = self.cross_attention(
             query=gen_emb, key=cli_emb, value=cli_emb, 
             attn_mask=img_mask, key_padding_mask=cli_mask)
        attn_output, _ = self.cross_attention2(
             query=img_emb, key=attn_output, value=attn_output)
        return attn_output

    
class encoder(nn.Module):
    def __init__(self, embed_dim=1024, depth=3, num_heads=16,
                 mlp_ratio=1., norm_layer=nn.LayerNorm, seq_len=5, learn_time_emb=False):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout=0., atten_drop=0.)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.learn_time_emb = learn_time_emb
        if self.learn_time_emb:
            self.time_emb = nn.Sequential(*[nn.Linear(1, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim), nn.ReLU()])


    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

    def forward(self, x, year, source_attn_mask=None, source_key_padding_mask=None):

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x,cls_tokens), dim=1)
        if self.learn_time_emb:
            x = x + self.time_emb(year.unsqueeze(-1)) # adding time embeding: (L+1) x D
        else:
            x = x + year.unsqueeze(-1) # adding time embeding: (L+1) x D

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, source_attn_mask, source_key_padding_mask)
        x = self.norm(x)
        cls_tokens = x[:,-1].squeeze()

        return cls_tokens

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, head_num, mlp_ratio, dropout, atten_drop):
        """
        args: input parameters to specify certain settings
        """
        # initialization
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.q_tran = nn.Linear(embed_dim,embed_dim)
        self.k_tran = nn.Linear(embed_dim,embed_dim)
        self.v_tran = nn.Linear(embed_dim,embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=head_num, dropout = atten_drop, batch_first=True)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for i in range(2)])
        encoder_ffn_embed_dim = int(self.embed_dim*mlp_ratio)

        self.fc1 = nn.Linear(self.embed_dim,encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(encoder_ffn_embed_dim,self.embed_dim)
    
    def forward(self, x, source_attn_mask=None, source_key_padding_mask=None):
        """
        Input:
            x (L*N*d tensor): input data. L is the target sequence length, N is the batch size and d is the embedding dimension.
            source_attn_mask (L*S tensor): This is the attention mask.
            source_key_padding_mask (N*S tensor): It is for the padding tokens.
        Output:
            x (L*N*d tensor): the processed data.
        """
        query = self.q_tran(x)
        key = self.k_tran(x)
        value = self.v_tran(x)

        residual = x
        x, _ = self.attn(query = query, key = key, value = value, key_padding_mask = source_key_padding_mask, attn_mask = source_attn_mask)
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = residual + x
        x = self.layer_norms[0](x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=self.dropout,training=self.training)
        x = self.fc2(x)
        x = F.dropout(x,p=self.dropout,training=self.training)
        # x = self.mlp(x)
        x = residual + x
        x = self.layer_norms[1](x)

        return x # L*N*d

class MultiheadAttention(nn.Module):  
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):   # add the bias parameter
        """
        embed_dim: an integer, Total dimension of the model.
        num_heads: an integer, Number of parallel attention heads.
        dropout: a real number between [0,1], Dropout probability on attn_weights. Default: 0.
        bias: a bool variable. If specified, adds bias to input / output projection layers. Default: True.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        # this is used to initialize the network parameters.
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)


    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Input:
            query (L*N*d tensor): Query embeddings. L is the target sequence length, N is the batch size and d is the embedding dimension.
            key (S*N*d tensor): Key embeddings. S is the source sequence length.
            value (S*N*d tensor): Value embeddings.
            key_padding_mask (N*S tensor): It is for the padding tokens.
            attn_mask (L*S tensor): This is the attention mask.
        Output:
            attn (L*N*d tensor): Attention outputs.
            attn_weights (d*d tensor): Attention weights averaged across heads.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        tgt_len, bsz, embed_dim = query.size()
        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        src_len = k.size(0)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
		
if __name__ == '__main__':

    v = TransformerWithClsToken(
        dim=128, heads=10, depth=6
    )

    time_series = torch.randn(4, 9, 128)
    logits = v(time_series) # (4, 1000)
    print(logits.shape)
            