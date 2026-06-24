import torch
from torch import nn

from cdi.bundled.leanbase.torch.select import select_activation


class MLPConditionEncoder(nn.Module):

    def __init__(
        self,
        input_shape,
        features,
        hidden_features = None,
        n_hidden        = 2,
        activ           = 'silu',
        n_tokens        = 1,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        if hidden_features is None:
            hidden_features = features

        input_features = 1
        for dim in input_shape:
            input_features *= dim

        layers = []
        curr_features = input_features

        for _ in range(n_hidden):
            layers.append(nn.Linear(curr_features, hidden_features))
            layers.append(select_activation(activ))
            curr_features = hidden_features

        layers.append(nn.Linear(curr_features, n_tokens * features))

        self.net = nn.Sequential(*layers)
        self._n_tokens = n_tokens
        self._features = features

    def forward(self, x):
        # x : (N, *input_shape)
        x = torch.flatten(x, start_dim = 1)
        x = self.net(x)
        return x.reshape(x.shape[0], self._n_tokens, self._features)
