import numpy as np
import torch
from pytorch_optimizer import SOAP

from ..models.nets import SumAdvantageNetwork
from .base import PDA_BASE


class PDA_DSC(PDA_BASE):
    """PDA for true discrete action spaces."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        args: object,
        device: str,
    ):
        super().__init__(obs_dim, act_dim, args, device, actor_approx=False)

        cfg_disc = getattr(args, "discretization", None)
        if cfg_disc is not None:
            raise ValueError(
                "PDA_DSC is discrete-only. Use PDA_BCD for discretized continuous action spaces."
            )

        self.num_act = self.act_dim
        self.act = torch.arange(self.num_act, device=self.device).unsqueeze(1).float()
        # For true discrete spaces, SumAdv consumes scalar action indices.
        self.SumAdv = SumAdvantageNetwork(
            obs_dim,
            1,
            hidden_sizes=self.hidden_sizes,
        ).to(self.device)
        self.optimSumAdv = SOAP(self.SumAdv.parameters(), lr=args.learning_rate)

    def compute_act(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        obs_t = self._obs_to_tensor(obs)
        return self.compute_act_direct(obs_t)

    def action_to_env(self, action: torch.Tensor) -> torch.Tensor:
        """Map internal action representation to environment actions."""
        if action.ndim > 1:
            action = torch.argmax(action, dim=-1)
        return action.long().view(-1)

    @torch.no_grad()
    def exploration_noise(self, act: torch.Tensor) -> torch.Tensor:
        prob = torch.as_tensor(self.act_prob, device=self.device)
        smoothing_factor = self.act_noise / (self.beta**0.3)
        prob.mul_(1 - smoothing_factor).add_(smoothing_factor / self.num_act)
        act_idx = torch.multinomial(prob, 1).squeeze(1)
        return act_idx.long()

    @torch.no_grad()
    def compute_act_direct(self, obs_t: torch.Tensor) -> torch.Tensor:
        batch_size = obs_t.shape[0]
        n_actions = self.num_act

        act = self.act.repeat(batch_size, 1)
        obs_expanded = obs_t.repeat_interleave(n_actions, dim=0)
        reg_factor = self.sum_beta / (self.beta**1.5 * self.reg)
        logits = self.SumAdv(obs_expanded, act) * reg_factor
        logits = logits.view(batch_size, n_actions)
        logits -= logits.max(dim=1, keepdim=True).values
        prob = torch.softmax(logits, dim=1)

        self.act_prob = prob
        act_opt = prob.argmax(dim=1)
        return act_opt.long()

    def optimize_sumAdv(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_adv: torch.Tensor,
        batch_returns: torch.Tensor,
        repeat: int,
        batch_size: int,
        maxgrad: float | None,
    ) -> list[float]:
        if batch_actions.ndim == 1:
            batch_actions = batch_actions.unsqueeze(-1)
        return super().optimize_sumAdv(
            batch_obs,
            batch_actions,
            batch_adv,
            batch_returns,
            repeat,
            batch_size,
            maxgrad,
        )


__all__ = ["PDA_DSC"]
