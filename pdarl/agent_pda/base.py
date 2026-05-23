from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from pytorch_optimizer import SOAP

from ..models.nets import ActorNetwork, SumAdvantageNetwork, ValueNetwork
from ..utils.rms import RunningMeanStd


class _ActorAdapter:
    """Adapter that keeps Trainer/Environment actor calls framework-compatible."""

    def __init__(self, compute_act_fn: Callable[[torch.Tensor | np.ndarray], torch.Tensor], act_lims: list[float]):
        self._compute_act_fn = compute_act_fn
        self._act_lims = act_lims

    def __call__(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self._compute_act_fn(obs)

    def map_action(self, action: torch.Tensor) -> torch.Tensor:
        act_low, act_high = self._act_lims[0], self._act_lims[1]
        return act_low + (action + 1.0) * (act_high - act_low) / 2.0


class PDA_BASE(nn.Module):
    """Own-framework base class for on-policy PDA variants."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        args: object,
        device: str,
        actor_approx: bool = True,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.beta = 1
        self.sum_beta = 1
        self.reg = args.step_size
        self.act_noise = args.act_noise
        self.adv_norm = args.adv_norm
        self.ret_norm = args.ret_norm
        self.recompute_ret = args.recompute_ret
        self.act_lims = args.act_lims
        self.hidden_sizes = args.hidden_sizes
        self.activation_name = args.activation
        self.activation = getattr(nn, self.activation_name)()
        self._eps = 1e-8

        if self.ret_norm:
            self.ret_rms = RunningMeanStd()

        self.V = ValueNetwork(obs_dim, hidden_sizes=self.hidden_sizes, activation=self.activation)
        self.SumAdv = SumAdvantageNetwork(obs_dim, act_dim, hidden_sizes=self.hidden_sizes, activation=self.activation)
        self.actor_model: Optional[nn.Module] = (
            ActorNetwork(
                obs_dim,
                act_dim,
                self.act_lims,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
            )
            if actor_approx
            else None
        )

        self.optimV = SOAP(self.V.parameters(), lr=args.learning_rate)
        self.optimSumAdv = SOAP(self.SumAdv.parameters(), lr=args.learning_rate)
        self.optimActor = SOAP(self.actor_model.parameters(), lr=args.learning_rate) if self.actor_model is not None else None

        # Keep Environment/Trainer contract untouched: agent.Actor(obs), agent.Actor.map_action(action).
        self.Actor = _ActorAdapter(self.compute_act, self.act_lims)

    def compute_act(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def _obs_to_tensor(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            return torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return obs.to(self.device, dtype=torch.float32)

    def action_to_env(self, action: torch.Tensor) -> torch.Tensor:
        """Map normalized action in [-1, 1] to environment action space."""
        return self.Actor.map_action(action)

    def exploration_noise(self, act: torch.Tensor) -> torch.Tensor:
        act_noise_magnitude = self.act_noise / (self.beta**0.3)
        act_noise_magnitude = max(act_noise_magnitude, 0.0)
        act = act + torch.randn_like(act) * act_noise_magnitude
        return torch.clamp(act, -1.0, 1.0)

    def optimize_V(
        self,
        batch_obs: torch.Tensor,
        batch_returns: torch.Tensor,
        repeat: int,
        batch_size: int,
        maxgrad: float | None,
    ) -> list[float]:
        losses: list[float] = []
        total_size = batch_obs.shape[0]
        split_batch_size = batch_size if batch_size else total_size

        for step in range(repeat):
            if self.recompute_ret and step > 0:
                _, batch_returns = self.compute_returns(self._rollout_data)
                batch_returns = batch_returns.reshape(-1)[self._b_valid]
            indices = np.random.permutation(total_size)
            for start in range(0, total_size, split_batch_size):
                end = min(start + split_batch_size, total_size)
                minibatch_indices = indices[start:end]
                minibatch_obs = batch_obs[minibatch_indices]
                minibatch_returns = batch_returns[minibatch_indices]

                value = self.V(minibatch_obs).flatten()
                loss = (minibatch_returns - value).pow(2).mean()
                self.optimV.zero_grad()
                loss.backward()
                if maxgrad:
                    nn.utils.clip_grad_norm_(self.V.parameters(), maxgrad)
                self.optimV.step()
                losses.append(loss.detach().item())
        return losses

    def _compute_sumadv_target(
        self,
        prev_sumadv: torch.Tensor,
        adv: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        return prev_sumadv * (1.0 - self.beta / self.sum_beta) + adv * self.beta / self.sum_beta

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
        losses: list[float] = []
        total_size = batch_obs.shape[0]
        split_batch_size = batch_size if batch_size else total_size

        if len(batch_actions.shape) == 1:
            batch_actions = batch_actions.unsqueeze(-1)

        previous_sumadv = self.SumAdv(batch_obs, batch_actions).flatten().detach()

        if self.adv_norm:
            mean, std = batch_adv.mean(), batch_adv.std()
            batch_adv_norm = (batch_adv - mean) / (std + self._eps)
        else:
            batch_adv_norm = batch_adv

        for _step in range(repeat):
            indices = np.random.permutation(total_size)
            for start in range(0, total_size, split_batch_size):
                end = min(start + split_batch_size, total_size)
                idx = indices[start:end]

                minibatch_obs = batch_obs[idx]
                minibatch_actions = batch_actions[idx]
                minibatch_adv = batch_adv_norm[idx]
                minibatch_returns = batch_returns[idx]
                minibatch_prev = previous_sumadv[idx]

                current_sumadv = self.SumAdv(minibatch_obs, minibatch_actions).flatten()
                target = self._compute_sumadv_target(minibatch_prev, minibatch_adv, minibatch_returns)
                err = current_sumadv - target
                loss = err.pow(2).mean()

                self.optimSumAdv.zero_grad()
                loss.backward()
                if maxgrad:
                    nn.utils.clip_grad_norm_(self.SumAdv.parameters(), maxgrad)
                self.optimSumAdv.step()
                losses.append(loss.detach().item())
        return losses

    def optimize_actor(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        repeat: int,
        batch_size: int,
        maxgrad: float | None,
    ) -> list[float]:
        del batch_obs, batch_actions, repeat, batch_size, maxgrad
        return [-1.0]

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

        if self.recompute_ret:
            advantages, returns = self.compute_returns(self._rollout_data)
            batch_adv = advantages.reshape(-1)[self._b_valid]
            batch_returns = returns.reshape(-1)[self._b_valid]

        sumadv_losses = self.optimize_sumAdv(
            batch_obs,
            batch_actions,
            batch_adv,
            batch_returns,
            self.args.update_epochs,
            self.args.minibatch_size,
            self.args.max_grad_norm,
        )

        actor_losses = self.optimize_actor(
            batch_obs,
            batch_actions,
            self.args.update_epochs,
            self.args.minibatch_size,
            self.args.max_grad_norm,
        )
        return v_losses, sumadv_losses, actor_losses

    def update_policy(
        self,
        rollout_data: dict,
        iteration: int,
        num_iterations: int,
    ) -> dict:
        obs = rollout_data["obs"]
        actions = rollout_data["actions"]
        init = rollout_data["init"]

        if self.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * self.args.learning_rate
            for optim in (self.optimV, self.optimSumAdv, self.optimActor):
                if optim is not None:
                    optim.param_groups[0]["lr"] = lrnow

        if self.recompute_ret:
            self._rollout_data = rollout_data.copy()

        advantages, returns = self.compute_returns(rollout_data)

        b_valid = (1.0 - init).reshape(-1).bool()
        self._b_valid = b_valid

        b_obs = obs.reshape((-1,) + obs.shape[2:])[b_valid]
        b_actions = actions.reshape((-1,) + actions.shape[2:])[b_valid]
        b_advantages = advantages.reshape(-1)[b_valid]
        b_returns = returns.reshape(-1)[b_valid]

        v_losses, sumadv_losses, actor_losses = self.learn_batches(
            b_obs, b_actions, b_advantages, b_returns
        )

        self.beta += 1
        self.sum_beta += self.beta

        v_loss = float(np.mean(v_losses)) if v_losses else 0.0
        sumadv_loss = float(np.mean(sumadv_losses)) if sumadv_losses else 0.0
        actor_loss = float(np.mean(actor_losses)) if actor_losses else -1.0

        with torch.no_grad():
            y_pred = self.V(b_obs).detach().reshape(-1).cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "v_loss": v_loss,
            "sumadv_loss": sumadv_loss,
            "actor_loss": actor_loss,
            "explained_var": explained_var,
            "learning_rate": self.optimV.param_groups[0]["lr"],
            "beta": self.beta,
        }

    @staticmethod
    def compute_gae(
        v_s: torch.Tensor,
        v_s_next: torch.Tensor,
        rew: torch.Tensor,
        end_flag: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> torch.Tensor:
        t_steps, n_envs = rew.shape
        adv = torch.zeros_like(rew)
        delta = rew + gamma * v_s_next - v_s
        disc = (1.0 - end_flag) * (gamma * lam)

        gae = torch.zeros(n_envs, dtype=rew.dtype, device=rew.device)
        for t in range(t_steps - 1, -1, -1):
            gae = delta[t] + disc[t] * gae
            adv[t] = gae
        return adv

    def compute_returns(self, rd: dict) -> tuple[torch.Tensor, torch.Tensor]:
        obs, rewards, dones = rd["obs"], rd["rewards"], rd["dones"]
        next_obs, terminated = rd["next_obs"], rd["terminated"]
        init = rd["init"]
        t_steps, n_envs = rewards.shape

        with torch.no_grad():
            flat_obs = obs.reshape(t_steps * n_envs, *obs.shape[2:])
            v_s = self.V(flat_obs).reshape(t_steps, n_envs)
            v_last = self.V(next_obs).reshape(n_envs)

        v_s_next = torch.cat([v_s[1:], v_last.unsqueeze(0)])

        if self.ret_norm:
            scale = torch.sqrt(torch.tensor(self.ret_rms.var + self._eps, device=v_s.device, dtype=v_s.dtype))
            v_s = v_s * scale
            v_s_next = v_s_next * scale

        v_s_next = v_s_next * (1.0 - terminated)

        end_flag = dones.clone()
        end_flag[-1] = 1.0

        adv = self.compute_gae(v_s, v_s_next, rewards, end_flag, self.args.gamma, self.args.gae_lambda)
        unnorm_ret = adv + v_s

        if self.ret_norm:
            scale = torch.sqrt(torch.tensor(self.ret_rms.var + self._eps, device=v_s.device, dtype=v_s.dtype))
            ret = unnorm_ret / scale
            valid = (1.0 - init).reshape(-1).bool()
            self.ret_rms.update(unnorm_ret.reshape(-1)[valid].cpu().numpy())
        else:
            ret = unnorm_ret

        return adv, ret


__all__ = ["PDA_BASE"]
