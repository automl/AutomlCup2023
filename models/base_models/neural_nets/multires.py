"""
Sequence Modeling with Multiresolution Convolutional Memory (ICML 2023)
by Jiaxin Shi, Ke Alexander Wang, Emily B. Fox
paper: https://arxiv.org/abs/2305.01638
GitHub: https://github.com/thjashin/multires-conv
"""


import torch
import torch.nn as nn


from .multireslayer import MultiresLayer


def masked_meanpool(x, lengths):
    # x: [bs, H, L]
    # lengths: [bs]
    L = x.shape[-1]
    # mask: [bs, L]
    mask = torch.arange(L, device=x.device) < lengths[:, None]
    # ret: [bs, H]
    return torch.sum(mask[:, None, :] * x, -1) / lengths[:, None]


class MultiresNet(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        batchnorm=False,
        encoder="embedding",
        n_tokens=None,
        layer_type="multires",
        max_length=None,
        depth=None,
        tree_select="fading",
        d_mem=None,
        kernel_size=2,
        indep_res_init=False,
    ):
        super().__init__()

        self.batchnorm = batchnorm
        self.max_length = max_length
        self.depth = depth
        if encoder == "linear":
            self.encoder = nn.Conv1d(d_input, d_model, 1)
        elif encoder == "embedding":
            self.encoder = nn.Embedding(n_tokens, d_model)
        self.activation = nn.GELU()

        # Stack sequence modeling layers as residual blocks
        self.seq_layers = nn.ModuleList()
        self.mixing_layers = nn.ModuleList()
        self.mixing_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        if batchnorm:
            norm_func = nn.BatchNorm1d
        else:
            norm_func = nn.LayerNorm

        for _ in range(n_layers):
            if layer_type == "multires":
                layer = MultiresLayer(
                    d_model,
                    kernel_size=kernel_size,
                    depth=depth,
                    tree_select=tree_select,
                    seq_len=max_length,
                    dropout=dropout,
                    memory_size=d_mem,
                    indep_res_init=indep_res_init,
                )
            else:
                raise NotImplementedError()
            self.seq_layers.append(layer)

            activation_scaling = 2
            mixing_layer = nn.Sequential(
                nn.Conv1d(d_model, activation_scaling * d_model, 1),
                nn.GLU(dim=-2),
                nn.Dropout1d(dropout),
            )

            self.mixing_layers.append(mixing_layer)
            self.norms.append(norm_func(d_model))

        # Linear layer maps to logits
        self.output_mapping = nn.Linear(d_model, d_output)

    def forward(self, x):
        """Input shape: [bs, d_input, seq_len]. """
        x = torch.squeeze(x.transpose(2, 4))
        is_zero = torch.any(x, dim=2) if len(x.shape) == 3 else x != 0
        lengths = torch.sum(is_zero, dim=1)

        if isinstance(self.encoder, nn.Embedding):
            x = self.encoder(x)
            x = x.transpose(-1, -2)
        elif isinstance(self.encoder, nn.Conv1d):
            x = x.transpose(-1, -2)
            x = self.encoder(x)
        # need to find the correct order!
        for layer, mixing_layer, norm in zip(
                self.seq_layers, self.mixing_layers, self.norms):
            x_orig = x
            x = layer(x)
            x = mixing_layer(x)
            x = x + x_orig

            x = apply_norm(x, norm, self.batchnorm)

        # mean_pooling: [bs, d_model, seq_len] -> [bs, d_model]
        if lengths is not None:
            # only pooling over the steps corresponding to actual inputs
            x = masked_meanpool(x, lengths)
        else:
            x = x.mean(dim=-1)

        # out: [bs, d_output]
        out = self.output_mapping(x)
        return out



def apply_norm(x, norm, batch_norm=False):
    if batch_norm:
        return norm(x)
    else:
        return norm(x.transpose(-1, -2)).transpose(-1, -2)