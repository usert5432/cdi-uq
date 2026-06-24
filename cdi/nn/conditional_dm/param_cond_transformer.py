import torch
from torch import nn

# Architecture adapted from JCDI-power-grid:
# https://github.com/fq123fq/JCDI-power-grid
# See cdi/nn/conditional_dm/NOTICE.md and LICENSE for attribution.

class TimeEmbeddingSin(nn.Module):

    def __init__(self, features, **kwargs):
        super().__init__(**kwargs)
        self._features = features

        self._sin_in = nn.Linear(1, features)
        self._out    = nn.Sequential(
            nn.Linear(features, features),
            nn.GELU(),
        )

    def forward(self, t):
        # t : (N, 1)
        result = self._sin_in(t)
        result = torch.sin(result)
        return self._out(result)

class ParamCondTransformer(nn.Module):

    def __init__(
        self, target_shape, features, features_ffn, features_cond,
        n_heads, n_layers, norm_first = True, time_embed = 'linear',
        **trans_kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        assert len(target_shape) == 1
        n_params = target_shape[0]

        self.cond_encoder = nn.Linear(features_cond, features)

        self.param_encoders = nn.ModuleList([
            nn.Linear(1, features) for _ in range(n_params)
        ])

        self.param_decoders = nn.ModuleList([
            nn.Linear(features, 1) for _ in range(n_params)
        ])

        if time_embed == 'sin':
            self.time_encoder = TimeEmbeddingSin(features)
        elif time_embed == 'linear':
            self.time_encoder = nn.Linear(1, features)
        else:
            raise ValueError(f"Unknown time embedding: '{time_embed}'")

        trans_layer = nn.TransformerEncoderLayer(
            d_model         = features,
            nhead           = n_heads,
            dim_feedforward = features_ffn,
            norm_first      = norm_first,
            batch_first     = True,
            **trans_kwargs
        )

        self.transformer = nn.TransformerEncoder(trans_layer, n_layers)

    def forward(self, x, cond, time):
        # x    : (N, n_params)
        # cond : (N, L_cond, features_cond)
        # time : (N, )

        # params : List[(N, 1)]
        params = x.split(1, dim = 1)

        # param_tokens : (N, n_params, features)
        param_tokens = torch.stack(
            [ enc(x) for (x, enc) in zip(params, self.param_encoders) ],
            dim = 1
        )

        # time_token : (N, 1, features)
        time       = time.unsqueeze(1).to(dtype = torch.float32)
        time_token = self.time_encoder(time).unsqueeze(1)

        # cond_tokens : (N, L_cond, features)
        cond_tokens = self.cond_encoder(cond)

        # in_tokens  : (N, n_params + 1 + L_cond, features)
        in_tokens = torch.cat([param_tokens, time_token, cond_tokens], dim = 1)

        # out_tokens  : (N, n_params + 1 + L_cond, features)
        out_tokens = self.transformer(in_tokens)

        # out_param_tokens  : (N, n_params, features)
        out_param_tokens = out_tokens[:, :len(params)]

        # out_param_token_list  : List[(N, features)]
        out_param_token_list = out_param_tokens.split(1, dim = 1)

        # out_params : List[(N, 1)]
        out_params = [
            dec(x).squeeze(2)
                for (x, dec) in zip(out_param_token_list, self.param_decoders)
        ]

        # result : (N, n_params)
        result = torch.cat(out_params, dim = 1)

        return result

class ParamImageCondTrans(ParamCondTransformer):

    def forward(self, x, cond, time):
        # x    : (N, n_params)
        # cond : (N, C, H, W)
        # time : (N, 1)

        (N, C, H, W) = cond.shape

        # cond : (N, C, H * W)
        cond = cond.reshape((N, C, H*W))

        # cond : (N, H * W, C) = (N, L, C)
        cond = cond.swapaxes(1, 2)

        return super().forward(x, cond, time)
