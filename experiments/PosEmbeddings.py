import torch
from torch import nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)



# pos =  PositionalEncoding(d_hid = 6,n_position=10)
# print(pos.pos_table)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, 1/freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def MyImplROPE(dim: int, end: int, theta: float = 10000.0):
    token_freqs = torch.pow(theta , -2*torch.arange(0,dim//2) / dim)
    # token_freqs = torch.repeat_interleave(torch.pow(theta , -2*torch.arange(0,dim//2) / dim),2)
    freq_scales= torch.arange(end)
    seq_frequencies  = torch.einsum("i,j->ij",freq_scales, token_freqs)
    freqs_cos = torch.cos(seq_frequencies)  # real part
    freqs_sin = torch.sin(seq_frequencies)  # imaginary part
    return freqs_cos, freqs_sin


val1 = precompute_freqs_cis(10,10)
val2 = MyImplROPE(10,10)

import pdb; pdb.set_trace()

