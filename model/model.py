import torch
import torch
from torch import nn
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ModelArgs:
    # default hyper parameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


class RMSNorm(nn.Module):
    """
    https://arxiv.org/abs/1910.07467
    Layer norm do Re-centring (mean = 0) and Re-scaling (1 variance ) But RMSNorm do just 
    Rescaling and it doesn't needs to depends on 2 properties like mean and variance.

    We don't need to calculate the mean. As hypothesis just convergence is due to variance, 

    RMSNorm:
    a_i = a_i/RMS * Weights
     
    Where in LayerNorm:
    needs to calculate mean and variance then make calculate layer norm : 
    
    Layer Norm would be 
    a_i = Norm(a_i)*y + b where w and b are learnable
    https://arxiv.org/abs/1607.06450
    """
    def __init__(self, eps, dim):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameters(dim)

    
    def __norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    
    def forward(self,x):
        output = self.__norm(x.float()).type_as(x)
        return output * self.weights



def get_rotatory_embeddings(seq_len, base_value):

    theta_arr  = torch.repeat_interleave(base_value**(2*torch.arange(0,seq_len//2 +1, device=DEVICE)/seq_len),2)
    coses = torch.cos(theta_arr).reshape(1,seq_len)
    sines = torch.sin(theta_arr).reshape(1,seq_len)
    return  coses , sines




def LLaMA(nn.Module):

    def __init__(self):
        self.rms_norm_in = RMSNorm(ModelArgs.dim, ModelArgs.eps)
        self.embedding_in = nn.Embedding(ModelArgs.vocab_size, ModelArgs.dim)



    def forward(self,tokens):
        bs, seq_len = tokens.shape
        in_emd = self.embedding_in(tokens)
        in_x = self.rms_norm_in(in_emd)

    


    
