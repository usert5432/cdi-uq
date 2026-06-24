import torch
from torch import nn

from cdi.bundled.leanbase.torch.select import select_activation


class MLPRegressor(nn.Module):

    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_features = 64,
        n_hidden        = 2,
        activ           = 'silu',
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        assert len(output_shape) == 1

        input_features = 1
        for dim in input_shape:
            input_features *= dim

        layers = []
        curr_features = input_features

        for _ in range(n_hidden):
            layers.append(nn.Linear(curr_features, hidden_features))
            layers.append(select_activation(activ))
            curr_features = hidden_features

        layers.append(nn.Linear(curr_features, output_shape[0]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        return self.net(x)
