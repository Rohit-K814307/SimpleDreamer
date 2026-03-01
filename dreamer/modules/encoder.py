import torch
import torch.nn as nn

from dreamer.utils.utils import (
    initialize_weights,
    horizontal_forward,
)


class Encoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.encoder

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Conv2d(
                self.observation_shape[0],
                self.config.depth * 1,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.Conv2d(
                self.config.depth * 1,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.Conv2d(
                self.config.depth * 2,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.Conv2d(
                self.config.depth * 4,
                self.config.depth * 8,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
        )
        self.network.apply(initialize_weights)

        # add the numeric state encodings
        self.state_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ELU(),
            nn.Linear(16, 8),
            nn.ELU()
        )
        self.state_net.apply(initialize_weights)

        self.projection_net = nn.Linear(1024 + 8, 1024)
        self.projection_net.apply(initialize_weights)


    def forward(self, x, x_s):
        img_embeddings = horizontal_forward(self.network, x, input_shape=self.observation_shape)
        img_embeddings = img_embeddings.reshape(img_embeddings.shape[0], img_embeddings.shape[1], -1)

        state_embeddings = horizontal_forward(self.state_net, x_s, input_shape=(3,))

        x = torch.cat([img_embeddings, state_embeddings], dim=-1)

        return horizontal_forward(self.projection_net, x, input_shape=(1024+8,))

