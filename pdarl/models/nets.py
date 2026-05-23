import torch
import torch.nn as nn


def _build_mlp(input_dim: int, output_dim: int, hidden_sizes: list[int], activation: nn.Module) -> nn.Sequential:
    """Build a tanh-activated MLP with configurable hidden sizes."""
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ValueNetwork(nn.Module):
    """Value network that estimates state values."""

    def __init__(self, obs_dim: int, hidden_sizes: list[int] = [64, 64], activation: nn.Module = nn.Tanh()):
        super().__init__()
        self.network = _build_mlp(obs_dim, 1, hidden_sizes, activation=activation)

    def forward(self, obs):
        return self.network(obs)


class SumAdvantageNetwork(nn.Module):
    """SumAdvantage network that takes observation and action as input."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int] = [64, 64], activation: nn.Module = nn.Tanh()):
        super().__init__()
        self.network = _build_mlp(obs_dim + act_dim, 1, hidden_sizes, activation=activation)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        return self.network(obs_action)


class ActorNetwork(nn.Module):
    """Actor network that outputs action mean directly."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_lims: list[float],
        hidden_sizes: list[int] = [64, 64],
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.network = _build_mlp(obs_dim, act_dim, hidden_sizes, activation=activation)
        self.network.append(activation)
        self.act_lims = act_lims

    def map_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map action from [-1, 1] to [act_low, act_high]."""
        act_low, act_high = self.act_lims[0], self.act_lims[1]
        return act_low + (action + 1.0) * (act_high - act_low) / 2.0

    def forward(self, obs):
        return self.network(obs)


class ProbActorNetwork(nn.Module):
    """Simple probabilistic actor with state-independent log-std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int] = [64, 64], activation: nn.Module = nn.Tanh()):
        super().__init__()
        self.mu_net = _build_mlp(obs_dim, act_dim, hidden_sizes, activation=activation)
        self.mu_net.append(nn.Tanh())
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.mu_net(obs)
        scale = torch.exp(self.log_std).expand_as(loc)
        return loc, scale
