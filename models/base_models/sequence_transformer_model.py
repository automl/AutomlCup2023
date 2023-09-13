"""
Reference : https://github.com/lucidrains/x-transformers
"""
import numpy as np
import torch
from ConfigSpace import EqualsCondition, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter, Constant
from skorch.callbacks import LRScheduler
from torch import nn
from torch.optim import AdamW, Adam
from x_transformers import Encoder, ContinuousTransformerWrapper

from .skorch_base import SkorchBaseModel


class SequenceTransformer(nn.Module):

    def __init__(self,
                 max_seq_len: int,
                 channel_in: int,
                 dim_out: int,
                 n_heads: int = 6,
                 depth: int = 10,
                 dim_attn: int = 128,
                 emb_dropout: float = 0.,
                 ff_swish: bool = False,
                 ff_relu_squared: bool = False,
                 layer_dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 ff_dropout: float = 0.1
                 ):
        super().__init__()
        # batch, sequence, channels
        self.transformer = ContinuousTransformerWrapper(
            dim_in=channel_in,  # maybe keep it at None then it won't apply a linear
            dim_out=None,
            max_seq_len=max_seq_len,
            emb_dropout=emb_dropout,
            attn_layers=Encoder(
                dim=dim_attn,
                depth=depth,
                heads=n_heads,
                use_rmsnorm=True,  # use_simple_rmsnorm for speed up
                ff_glu=True,
                ff_swish=ff_swish,
                ff_relu_squared=ff_relu_squared,
                rel_pos_bias=True,
                layer_dropout=layer_dropout,  # stochastic depth - dropout entire layer
                attn_dropout=attn_dropout,  # dropout post-attention
                ff_dropout=ff_dropout,
            )
        )
        self.output_layer = nn.Linear(max_seq_len * dim_attn, dim_out)

    def forward(self, x):
        x = self.transformer(x.squeeze())
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.output_layer(x)
        return x


class SequenceTransformer_Model(SkorchBaseModel):

    def __init__(self, metadata, input_cfg=None, seed=None):
        super().__init__(metadata, input_cfg, seed)

        if self.cfg['ff_activation'] == 'ff_relu_squared':
            self.skorch_kwargs['module__ff_relu_squared'] = True
        elif self.cfg['ff_activation'] == 'ff_swish':
            self.skorch_kwargs['module__ff_swish'] = True

        if self.cfg['optimizer'] == 'Adam':
            self.skorch_kwargs['optimizer'] = Adam
        elif self.cfg['optimizer'] == 'AdamW':
            self.skorch_kwargs['optimizer'] = AdamW
            self.skorch_kwargs['optimizer__weight_decay'] = self.cfg['weight_decay']
        else:
            raise ValueError(f"Unknown value {self.cfg['optimizer']} for optimizer")

        if self.cfg['lr_scheduler']:
            self.callbacks.append(
                self.callbacks.append(LRScheduler(policy='CosineAnnealingLR', T_max=self.cfg['max_epochs'])))

        # stratify = False
        self.model = self.model_class(
            SequenceTransformer,
            max_epochs=self.cfg['max_epochs'],
            module__max_seq_len=self.metadata.input_shape.max_sequence_len,
            module__channel_in=self.metadata.input_shape.channels,
            module__dim_out=np.prod(self.metadata.output_shape),
            module__n_heads=self.cfg['n_heads'],
            module__depth=self.cfg['depth'],
            module__dim_attn=self.cfg['dim_attn'],
            module__emb_dropout=self.cfg['emb_dropout'],
            lr=self.cfg['lr'],
            callbacks=self.callbacks,
            **self.skorch_kwargs
        )

    @classmethod
    def get_config_space(cls):
        cs = super().get_config_space()
        name = Constant(name="model_name", value="seq_transformer")

        lr = UniformFloatHyperparameter(
            name="lr", lower=1e-5, upper=1e-2, default_value=1e-3, log=True
        )
        optimizer = CategoricalHyperparameter(
            name="optimizer", choices=['Adam', 'AdamW'], default_value='AdamW'
        )
        lr_scheduler = CategoricalHyperparameter(
            name="lr_scheduler", choices=[True, False], default_value=True
        )
        weight_decay = UniformFloatHyperparameter(
            name="weight_decay", lower=1e-5, upper=1e-1, default_value=0.03, log=True
        )
        max_epochs = UniformIntegerHyperparameter(
            name="max_epochs", lower=1, upper=200, default_value=100, log=False
        )
        n_heads = UniformIntegerHyperparameter(
            name="n_heads", lower=4, upper=8, default_value=4, log=False
        )
        depth = UniformIntegerHyperparameter(  # also number of layers
            name="depth", lower=1, upper=10, default_value=6, log=False
        )
        dim_attn = UniformIntegerHyperparameter(
            name="dim_attn", lower=1, upper=256, default_value=64, log=False
        )
        layer_dropout = UniformFloatHyperparameter(
            name="layer_dropout", lower=0., upper=0.5, default_value=0.2, log=False
        )
        attn_dropout = UniformFloatHyperparameter(
            name="attn_dropout", lower=0., upper=0.5, default_value=0.2, log=False
        )
        ff_dropout = UniformFloatHyperparameter(
            name="ff_dropout", lower=0., upper=0.5, default_value=0.2, log=False
        )
        emb_dropout = UniformFloatHyperparameter(
            name="emb_dropout", lower=0., upper=0.5, default_value=0.2, log=False
        )
        ff_activation = CategoricalHyperparameter(
            name="ff_activation", choices=['ff_relu_squared', 'ff_swish', 'ff_gelu'], default_value='ff_gelu'
        )

        cs.add_hyperparameters(
            [name, lr, weight_decay, max_epochs, n_heads, depth, dim_attn, emb_dropout, ff_activation, layer_dropout,
             attn_dropout, ff_dropout, optimizer, lr_scheduler])
        weight_decay_condition = EqualsCondition(weight_decay, optimizer, 'AdamW')
        cs.add_conditions([weight_decay_condition])
        return cs

    @classmethod
    def is_applicable(cls, metadata):
        return metadata.input_dimension == 1 and metadata.input_shape.max_sequence_len < 1000
