import itertools
from typing import Optional

import numpy as np
import torch

from .base import PDA_BASE


class PDA_OPT(PDA_BASE):
    """PDA with direct action-subproblem optimization via RGD or ACFGM."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        args: object,
        device: str,
        action_opt: str = "RGD",
        action_opt_itr: int = 75,
        action_opt_params: Optional[list[float] | float] = None,
        actor_approx: bool = True,
    ):
        super().__init__(obs_dim, act_dim, args, device, actor_approx=actor_approx)

        self.action_opt_itr = int(getattr(args, "action_opt_itr", action_opt_itr))
        self.action_opt = str(getattr(args, "action_opt", action_opt)).upper()
        cfg_params = getattr(args, "action_opt_params", action_opt_params)

        if self.action_opt == "ACFGM":
            from ..utils.acfgm import ACFGM

            self.acfgm_beta = float(cfg_params) if cfg_params is not None else 0.3
            self.ACFGM = ACFGM
            self.compute_act_fn = self.compute_act_acfgm
        else:
            from ..utils.rgd import RGD

            default_params = [1e-4, 1e-2]
            if cfg_params is None:
                cfg_params = default_params
            self.rgd_mu = float(cfg_params[0])
            self.rgd_lr = float(cfg_params[1])
            self.RGD = RGD
            self.compute_act_fn = self.compute_act_rgd

    def compute_act(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        obs_t = self._obs_to_tensor(obs)
        if self.actor_model is not None and (self.beta % 2 == 0):
            return self.actor_model(obs_t).detach()
        return self.compute_act_fn(obs_t)

    def compute_act_rgd(self, obs_t: torch.Tensor) -> torch.Tensor:
        act_opt = torch.zeros(obs_t.shape[0], self.act_dim, device=self.device, requires_grad=True)
        mu = self.rgd_mu
        lr = self.rgd_lr
        optimizer_act = self.RGD([act_opt], lr=lr, lims=[-1.0, 1.0], momentum=0.9, nesterov=True)

        for _ in range(self.action_opt_itr):

            def closure() -> torch.Tensor:
                optimizer_act.zero_grad()
                u = torch.randn(act_opt.shape, device=self.device)
                u = u / torch.linalg.norm(u, dim=1, keepdim=True)
                act_opt1 = act_opt + mu * u
                act_pair = torch.concat([act_opt, act_opt1], dim=0)
                obs_pair = obs_t.repeat(2, 1)
                out_pair = (
                    -self.SumAdv(obs_pair, act_pair)
                    + (self.beta**1.5 * self.reg * torch.sum(act_pair**2, dim=1, keepdim=True)) / self.sum_beta
                )
                out, out1 = torch.split(out_pair, len(act_opt))
                grad = (out1 - out) / mu * u
                act_opt.grad = torch.nan_to_num(grad, nan=1e-9)
                return out

            optimizer_act.step(closure)

        return torch.clamp(act_opt.detach(), -1.0, 1.0)

    def compute_act_acfgm(self, obs_t: torch.Tensor) -> torch.Tensor:
        act_opt = torch.zeros(obs_t.shape[0], self.act_dim, device=self.device, requires_grad=True)
        optimizer_act = self.ACFGM([act_opt], beta=self.acfgm_beta, lims=[-1.0, 1.0])
        vec = torch.ones(obs_t.shape[0], 1, device=self.device)

        for _ in range(self.action_opt_itr):

            def closure() -> torch.Tensor:
                optimizer_act.zero_grad()
                out = (
                    -self.SumAdv(obs_t, act_opt)
                    + (self.beta**1.5 * self.reg * torch.sum(act_opt**2, dim=1, keepdim=True)) / self.sum_beta
                )
                out.backward(gradient=vec, retain_graph=True)
                return out

            optimizer_act.step(closure)

        return torch.clamp(act_opt.detach(), -1.0, 1.0)

    def optimize_actor(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        repeat: int,
        batch_size: int,
        maxgrad: float | None,
    ) -> list[float]:
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

                current_a = self.actor_model(batch_obs[idx])
                target_a = batch_actions[idx]
                err = current_a - target_a
                actor_loss = err.pow(2).mean()

                self.optimActor.zero_grad()
                actor_loss.backward()
                if maxgrad:
                    torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), maxgrad)
                self.optimActor.step()
                losses.append(actor_loss.detach().item())

        return losses

    def learn_batches(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_adv: torch.Tensor,
        batch_returns: torch.Tensor,
    ) -> tuple[list[float], list[float], list[float]]:
        v_losses = self.optimize_V(
            batch_obs,
            batch_returns,
            self.args.update_epochs,
            self.args.minibatch_size,
            self.args.max_grad_norm,
        )

        sumadv_losses = self.optimize_sumAdv(
            batch_obs,
            batch_actions,
            batch_adv,
            batch_returns,
            self.args.update_epochs,
            self.args.minibatch_size,
            self.args.max_grad_norm,
        )

        if self.actor_model is not None and (self.beta % 2 == 1):
            actor_losses = self.optimize_actor(
                batch_obs,
                batch_actions,
                self.args.update_epochs,
                self.args.minibatch_size,
                self.args.max_grad_norm,
            )
        else:
            actor_losses = [-1.0]

        return v_losses, sumadv_losses, actor_losses


class PDA_BCD(PDA_BASE):
    """PDA for continuous control via discretized block-coordinate descent."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        args: object,
        device: str,
    ):
        super().__init__(obs_dim, act_dim, args, device, actor_approx=False)
        if self.act_lims is None:
            raise ValueError(
                "PDA_BCD requires continuous action limits; use PDA_DSC for discrete action spaces."
            )

        cfg_disc = getattr(args, "discretization", None)
        if cfg_disc is None:
            raise ValueError(
                "PDA_BCD requires a positive discretization value for continuous action spaces."
            )

        cfg_action_opt_itr = getattr(args, "action_opt_itr", None)
        if cfg_action_opt_itr is None:
            raise ValueError(
                "PDA_BCD requires action_opt_itr to be provided."
            )
        self.action_opt_itr = int(cfg_action_opt_itr)
        if self.action_opt_itr <= 0:
            raise ValueError("PDA_BCD action_opt_itr must be greater than 0.")

        self.discretization = int(cfg_disc)
        if self.discretization <= 1:
            raise ValueError("PDA_BCD discretization must be greater than 1.")

        self.num_act = self.discretization
        self.act_base = torch.linspace(
            self.act_lims[0], self.act_lims[1], self.discretization, device=self.device
        )
        self.act = (
            torch.ones([self.discretization, self.act_dim], device=self.device)
            * self.act_base[self.discretization // 2]
        )
        self.act_idx_iter = itertools.cycle(np.arange(self.act_dim))

    def compute_act(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        obs_t = self._obs_to_tensor(obs)
        return self.compute_act_bcd(obs_t)

    def action_to_env(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action, self.act_lims[0], self.act_lims[1])

    @torch.no_grad()
    def exploration_noise(self, act: torch.Tensor) -> torch.Tensor:
        del act
        prob = torch.as_tensor(self.act_prob, device=self.device)
        smoothing_factor = self.act_noise / (self.beta**0.3)
        prob.mul_(1 - smoothing_factor).add_(smoothing_factor / self.num_act)
        bsz, act_dim, disc = prob.shape
        prob_flat = prob.reshape(-1, disc)
        act_idx = torch.multinomial(prob_flat, 1).reshape(bsz, act_dim)
        return self.act_base[act_idx]

    @torch.no_grad()
    def compute_act_bcd(self, obs_t: torch.Tensor) -> torch.Tensor:
        batch_size = obs_t.shape[0]
        n_disc = self.num_act
        act_dim = self.act_dim

        act = self.act.clone().repeat(batch_size, 1)
        obs_expanded = obs_t.repeat_interleave(n_disc, dim=0)
        act_prob = torch.ones((batch_size, act_dim, n_disc), device=self.device)

        reg_factor = self.sum_beta / (self.beta**1.5 * self.reg)

        for _ in range(self.action_opt_itr):
            act_idx = next(self.act_idx_iter)
            act[:, act_idx] = self.act_base.repeat(batch_size)
            logits = self.SumAdv(obs_expanded, act) * reg_factor
            logits = logits.view(batch_size, n_disc)
            logits -= logits.max(dim=1, keepdim=True).values
            prob = torch.softmax(logits, dim=1)
            act_prob[:, act_idx, :] = prob
            best_idx = prob.argmax(dim=1)
            act[:, act_idx] = self.act_base[best_idx].repeat_interleave(n_disc)

        self.act_prob = act_prob
        act_opt_idx = act_prob.argmax(dim=2)
        return self.act_base[act_opt_idx]


__all__ = ["PDA_OPT", "PDA_BCD"]
