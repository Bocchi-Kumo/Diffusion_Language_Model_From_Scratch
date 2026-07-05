import torch
from torch import nn
from einops import einsum, rearrange
from basics import nn_utils
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        
        # Initialize weights with truncated normal distribution
        mean = 0.0
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: size of the vocabulary
        embedding_dim: dimension of each embedding vector
        '''
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        # Initialize weights with truncated normal distribution
        mean = 0.0
        std = 1
        nn.init.trunc_normal_(self.embedding_matrix, mean=mean, std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[x]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        Root Mean Square Layer Normalization
        -----------
        d_model: Hidden dimension of the model
        eps: Epsilon value for numerical stability
        '''
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) # Learnable gain parameter and initialized to ones

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = (x / RMS) * self.gain
        return result.to(in_dtype)

def SiLU(x:torch.Tensor) -> torch.Tensor:    
    return x * torch.sigmoid(x)

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4          # common default
        self.mlp = nn.Sequential(
            Linear(d_model, d_ff),
            nn.SiLU(),
            Linear(d_ff, d_model),
        )
        self.half_dim = d_model // 2
        self.freqs = math.log(10000) / (self.half_dim - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
            """
            t: shape [batch_size] or [batch_size, 1] → always returns clean [batch_size, d_model]
            """
            t = t.float()
            # Robustly squeeze any trailing singleton dimensions (handles [B,1] safely)
            while t.dim() > 1 and t.size(-1) == 1:
                t = t.squeeze(-1)
            if t.dim() > 1:
                t = t.view(-1)

            # Correct sinusoidal frequencies
            freqs = torch.exp(-torch.arange(self.half_dim, device=t.device) * self.freqs)
            emb = t.unsqueeze(-1) * freqs.unsqueeze(0)   # [B, half_dim]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [batch_size, d_model]

            return self.mlp(emb)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

        # Initialize weights with truncated normal distribution
        mean = 0.0
        std = (2 / (d_model + d_ff)) ** 0.5
        nn.init.trunc_normal_(self.w1, mean=mean, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w2, mean=mean, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w3, mean=mean, std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w1x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3x = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2, SiLU(w1x) * w3x, "d_model d_ff, ... d_ff -> ... d_model")
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        theta: value used to compute the rotation angles
        d_k: dimension of the key/query vectors
        max_seq_len: maximum sequence length that will be inputted
        device: device to store the buffer on
        '''
        super().__init__()
        self.pair_count = d_k // 2
        cos_table = torch.zeros((max_seq_len, d_k // 2), device=device)
        sin_table = torch.zeros((max_seq_len, d_k // 2), device=device)
        for i in range(0, max_seq_len):
            for j in range(0, d_k // 2):
                angle = i / (theta ** ((2 * j) / d_k))
                cos_table[i, j] = torch.cos(torch.tensor(angle))
                sin_table[i, j] = torch.sin(torch.tensor(angle))
        self.register_buffer("cos_table", cos_table) # (max_seq_len, pair_count(d_k/2))
        self.register_buffer("sin_table", sin_table) # (max_seq_len, pair_count(d_k/2))
    
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor = None) -> torch.Tensor:
        '''
        x: Float[Tensor, "... seq_length d_k"]: input query or key tensor
        token_positions: Int[Tensor, "... seq_length"]: positions of the tokens in the sequence
        '''
        x_pairs = rearrange(x, '... seq_length (pair_count pair) -> ... seq_length pair_count pair', pair=2) # shape [..., seq_length, pair_count, 2]
        cos = self.cos_table[token_positions]  # shape [..., seq_length, pair_count]
        sin = self.sin_table[token_positions]  # shape [..., seq_length, pair_count]

        out1 = x_pairs[..., 0] * cos - x_pairs[..., 1] * sin
        out2 = x_pairs[..., 0] * sin + x_pairs[..., 1] * cos
        output = torch.stack([out1, out2], dim=-1)
        return rearrange(output, '... pair_count pair -> ... (pair_count pair)')     
    
def Attention(Q, K, V, mask=None):
    '''
    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    '''
    scores = (einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (Q.size(-1) ** 0.5)).to(torch.float32)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e6)
    softmax_scores = (nn_utils.softmax(scores, dim=-1))
    return (einsum(softmax_scores, V.to(torch.float32), "... queries keys, ... keys d_v -> ... queries d_v")).to(Q.dtype)

class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta : int = None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        d_head = d_model // num_heads
        self.d_head = d_head
        self.QKV_weights = nn.Parameter(torch.empty((3*d_model, d_model), device=device, dtype=dtype))
        self.O_weights = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        if theta is not None:
            self.RoPE = RoPE(theta=theta, d_k=d_head, max_seq_len=max_seq_len, device=device)
        else:
            self.RoPE = None
        # Initialize weights with truncated normal distribution
        mean = 0.0
        std = (2 / (d_model + d_model)) ** 0.5
        nn.init.trunc_normal_(self.QKV_weights, mean=mean, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.O_weights, mean=mean, std=std, a=-3*std, b=3*std)

    def forward(self, X:torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_length = X.size(-2)
        # Projecting input X to Q, K, V
        QKV_projection = einsum(self.QKV_weights, X, "out_features in_features, ... seq_length in_features -> ... seq_length out_features") # out_features = 3*d_model
        Q, K, V = torch.split(QKV_projection, self.d_model, dim=-1) # Each of shape (..., seq_length, d_model)
        Q = rearrange(Q, "... seq_length (num_heads d_head) -> ... num_heads seq_length d_head", num_heads=self.num_heads, d_head=self.d_head)
        K = rearrange(K, "... seq_length (num_heads d_head) -> ... num_heads seq_length d_head", num_heads=self.num_heads, d_head=self.d_head)
        V = rearrange(V, "... seq_length (num_heads d_head) -> ... num_heads seq_length d_head", num_heads=self.num_heads, d_head=self.d_head)

        # Applying RoPE on Q, K
        if self.RoPE is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_length, device=X.device)  # [seq_length]
                expand_shape = [1] * (Q.ndim - 2) + [seq_length]
                token_positions = token_positions.reshape(expand_shape)  # shape: [1, ..., 1, seq_length]
                # Expand to match Q's leading dimensions and num_heads
                expand_to = list(Q.shape[:-1])  # all dims except d_head
                token_positions = token_positions.expand(*expand_to)  # shape: [..., num_heads, seq_length]
                Q = self.RoPE(Q, token_positions)
                K = self.RoPE(K, token_positions)
            else:
                Q = self.RoPE(Q, token_positions)
                K = self.RoPE(K, token_positions)
        else: # Not applying RoPE
            pass

        # For a diffusion language model we want the language model could see the whole tensor.
        Attentions = Attention(Q, K, V) 
        Attentions = rearrange(Attentions, '... num_heads seq_length d_head -> ... seq_length (num_heads d_head)') # Concat heads
        MultiHeadSelfAttention = einsum(self.O_weights, Attentions, "out_features in_features, ... seq_length in_features -> ... seq_length out_features")

        return MultiHeadSelfAttention
    
class transformer_block_timestep(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: int = None):
        super().__init__()
        self.multihead_self_attention = multihead_self_attention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.RMSNorm1 = RMSNorm(d_model=d_model) # First RMSNorm layer for self-attention
        self.RMSNorm2 = RMSNorm(d_model=d_model) # Second RMSNorm layer for feed-forward
        self.SwiGLU = SwiGLU(d_model=d_model, d_ff=d_ff) # Feed-forward layer
        self.TimestepEmbedding = TimestepEmbedding(d_model=d_model, d_ff=d_ff)

        #===ADEDE===
        self.adaLN_scale = Linear(d_model, d_model)
        self.adaLN_shift = Linear(d_model, d_model)

    def forward(self, x:torch.Tensor, t:torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
            # Get time embedding (now always returns clean [B, d_model])
            time_embed = self.TimestepEmbedding(t)
            if time_embed.dim() == 2:
                time_embed = time_embed.unsqueeze(1)                  # [B, 1, D]
            elif time_embed.dim() == 3 and time_embed.size(1) != 1:
                time_embed = time_embed.unsqueeze(1)

            # Predict scale and shift from time embedding
            scale = self.adaLN_scale(time_embed)                      # [batch_size, 1, d_model]
            shift = self.adaLN_shift(time_embed)                      # [batch_size, 1, d_model]

            # Self-attention with adaLN
            normed = self.RMSNorm1(x)
            normed = normed * (1 + scale) + shift                     # adaLN
            attn_out = self.multihead_self_attention(normed, token_positions=token_positions)
            x = x + attn_out

            # Feed-forward with adaLN
            normed = self.RMSNorm2(x)
            normed = normed * (1 + scale) + shift                     # adaLN
            ffn_out = self.SwiGLU(normed)
            x = x + ffn_out
            return x

class diffusion_lm(nn.Module):
    def __init__(self, vocab_size: int, context_length:int , d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: int):
        super().__init__()
        self.embedding_layer = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = nn.ModuleList([
            transformer_block_timestep(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta)
            for _ in range(num_layers)])
        self.RMSNorm = RMSNorm(d_model=d_model)
        self.output_layer = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x:torch.Tensor, t:torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        '''
        t should get shape like [batch_size, 1] and normalized between [0, 1]
        '''
        # t = t.squeeze(-1)
        if token_positions is None:
            seq_length = x.size(-1)
            token_positions = torch.arange(seq_length, device=x.device)

        x = self.embedding_layer(x)

        for block in self.transformer_blocks:
            x = block(x, t, token_positions=token_positions)

        x = self.RMSNorm(x)
        y = self.output_layer(x)
        return y