from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from .base import PDA_BASE


class PDA_ACT(PDA_BASE):
    """PDA with actor-gradient subproblem solve (current own-framework baseline)."""

    def __init__(self, obs_dim: int, act_dim: int, args: object, device: str):
        super().__init__(obs_dim, act_dim, args, device, actor_approx=True)

    def compute_act(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        if self.actor_model is None:
            raise RuntimeError("Actor model is required for PDA_ACT")
        obs_t = self._obs_to_tensor(obs)
        return self.actor_model(obs_t)

    def optimize_actor(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        repeat: int,
        batch_size: int,
        maxgrad: float | None,
    ) -> list[float]:
        del batch_actions
        if self.actor_model is None or self.optimActor is None:
            return [-1.0]

        losses: list[float] = []
        total_size = batch_obs.shape[0]
        split_batch_size = batch_size if batch_size else total_size

        for _step in range(repeat):
            indices = np.random.permutation(total_size)
            for start in range(0, total_size, split_batch_size):
                end = min(start + split_batch_size, total_size)
                idx = indices[start:end]
                minibatch_obs = batch_obs[idx]

                current_a = self.actor_model(minibatch_obs)
                regularization = (
                    self.beta**1.5 * self.reg * torch.sum(current_a**2, dim=1, keepdim=True)
                ) / self.sum_beta
                actor_loss = -self.SumAdv(minibatch_obs, current_a) + regularization
                actor_loss = actor_loss.mean()

                self.optimActor.zero_grad()
                actor_loss.backward()
                if maxgrad:
                    nn.utils.clip_grad_norm_(self.actor_model.parameters(), maxgrad)
                self.optimActor.step()
                losses.append(actor_loss.detach().item())

        return losses


class PMD_ACT(PDA_ACT):
    """Policy Mirror Descent variant with old-policy regularization."""

    def __init__(self, obs_dim: int, act_dim: int, args: object, device: str):
        super().__init__(obs_dim, act_dim, args, device)
        if self.actor_model is None:
            raise RuntimeError("PMD_ACT requires an actor model")
        self.Actor_old = deepcopy(self.actor_model)
        self.Actor_old.eval()

    def _compute_sumadv_target(
        self,
        prev_sumadv: torch.Tensor,
        adv: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        return adv

    def optimize_actor(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        repeat: int,
        batch_size: int,
        maxgrad: float | None,
    ) -> list[float]:
        del batch_actions
        if self.actor_model is None or self.optimActor is None:
            return [-1.0]

        losses: list[float] = []
        total_size = batch_obs.shape[0]
        split_batch_size = batch_size if batch_size else total_size

        with torch.no_grad():
            old_act = self.Actor_old(batch_obs)

        for _step in range(repeat):
            indices = np.random.permutation(total_size)
            for start in range(0, total_size, split_batch_size):
                end = min(start + split_batch_size, total_size)
                idx = indices[start:end]

                minibatch_obs = batch_obs[idx]
                minibatch_old_act = old_act[idx]
                current_a = self.actor_model(minibatch_obs)

                regularization = self.beta * self.reg * torch.sum(
                    (current_a - minibatch_old_act) ** 2,
                    dim=1,
                    keepdim=True,
                )
                actor_loss = -self.SumAdv(minibatch_obs, current_a) + regularization
                actor_loss = actor_loss.mean()

                self.optimActor.zero_grad()
                actor_loss.backward()
                if maxgrad:
                    nn.utils.clip_grad_norm_(self.actor_model.parameters(), maxgrad)
                self.optimActor.step()
                losses.append(actor_loss.detach().item())

        self.Actor_old.load_state_dict(self.actor_model.state_dict())
        return losses


__all__ = ["PDA_ACT", "PMD_ACT"]
