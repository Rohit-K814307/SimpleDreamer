import torch
import torch.nn as nn

from dreamer.utils.utils import (
    initialize_weights,
    horizontal_forward,
    create_normal_dist,
    build_network,
)


class Decoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Linear(
                self.deterministic_size + self.stochastic_size, self.config.depth * 32
            ),
            nn.Unflatten(1, (self.config.depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(
                self.config.depth * 32,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 4,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 2,
                self.config.depth * 1,
                self.config.kernel_size + 1,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 1,
                self.observation_shape[0],
                self.config.kernel_size + 1,
                self.config.stride,
            ),
        )
        self.network.apply(initialize_weights)

        # Delta prediction head: (posterior + deterministic) → 16-dim delta targets
        # d_pos_quad(3) + d_vel_quad(3) + d_pos_payload(3) + d_quat(4) + d_vel_payload(3)
        self.delta_net = build_network(
            self.deterministic_size + self.stochastic_size,
            self.config.delta_hidden_size,
            self.config.delta_num_layers,
            self.config.activation,
            16,
        )
        self.delta_net.apply(initialize_weights)

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=self.observation_shape
        )
        dist = torch.distributions.Independent(
            torch.distributions.Laplace(x, torch.ones_like(x)),
            len(self.observation_shape),
        )
        return dist

    def forward_delta(self, posterior, deterministic):
        x = horizontal_forward(
            self.delta_net, posterior, deterministic, output_shape=(16,)
        )
        return create_normal_dist(x, std=1, event_shape=1)



