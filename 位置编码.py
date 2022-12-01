from functools import partial
import torch
import numpy as np
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

pos_embed = nn.Parameter(torch.rand(1,196,1024))
patch_num = 196
bed_dim = 1024
def pos_coding(patch_num, bed_dim):
    patch = int(patch_num**0.5)
    a = np.arange(patch).reshape(1,patch)
    b = np.arange(patch).reshape(1,patch)
    bed_dim_omega = bed_dim//4
    omega = 1/(10000**((np.arange(bed_dim_omega).reshape(-1))/256.))
    out1 = np.einsum('m,d->md',np.stack(np.meshgrid(a,b)).reshape(2,1,patch,patch)[0].reshape(-1),omega)
    out1_sin_cos = np.concatenate([np.sin(out1),np.cos(out1)],axis=1)
    out2 = np.einsum('m,d->md',np.stack(np.meshgrid(a,b)).reshape(2,1,patch,patch)[1].reshape(-1),omega)
    out2_sin_cos = np.concatenate([np.sin(out2),np.cos(out2)],axis=1)
    output = np.concatenate([out1_sin_cos,out2_sin_cos],axis=1)
    return output

pos = pos_coding(patch_num,bed_dim)
pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))