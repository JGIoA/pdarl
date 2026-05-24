# Single-file implementation of PDA_DSC in the CleanRL style.

import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from pytorch_optimizer import SOAP
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cpu"
    """compute device: cpu, cuda or mps"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PDA"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    logging: bool = True
    """whether to log training/testing statistics to tensorboard"""

    # Algorithm specific arguments (defaults match config/pda_dsc.yaml)
    env_id: str = "LunarLander-v3"
    """the id of the environment"""
    env_kargs: Optional[dict] = None
    """extra kwargs forwarded to gym.make"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the SOAP optimizers (V, SumAdv)"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_test_envs: int = 10
    """the number of parallel test environments"""
    test_interval: int = 5
    """run evaluation every N iterations"""
    num_steps: int = 400
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """toggle learning rate annealing for V/SumAdv optimizers"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the generalized advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update each of V and SumAdv per iteration"""
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    """hidden layer sizes for value/sum-advantage MLPs"""
    activation: str = "Tanh"
    """activation function name (resolved against torch.nn)"""
    obs_norm: bool = True
    """whether to normalize observations with a running mean/std"""
    max_grad_norm: Optional[float] = 0.5
    """the maximum norm for the gradient clipping (None to disable)"""
    ret_norm: bool = True
    """whether to scale returns by a running standard deviation"""
    adv_norm: bool = True
    """whether to standardize advantages inside the SumAdv update"""
    recompute_ret: bool = True
    """whether to recompute returns inside V optimization and between V/SumAdv"""

    # PDA_DSC-specific
    step_size: float = 0.9
    """regularization coefficient (step size) for PDA"""
    act_noise: float = 0.5
    """exploration smoothing strength for the discrete policy"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def _make_single_env(env_id: str, seed: int, env_kargs: Optional[dict]):
    """Construct a single env and seed its ``np_random``."""
    env = gym.make(env_id, **(env_kargs or {}))
    rng = np.random.default_rng(seed=seed)
    env.np_random = rng
    if hasattr(env, "_np_random_seed"):
        env._np_random_seed = int(seed)
    return env


def make_vector_env(
    env_id: str,
    num_envs: int,
    seeds: list[int],
    obs_norm: bool,
    test_obs_rms: object = None,
    env_kargs: Optional[dict] = None,
):
    envs = gym.vector.SyncVectorEnv(
        [
            (lambda s=seeds[i]: _make_single_env(env_id, s, env_kargs))
            for i in range(num_envs)
        ],
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    if isinstance(envs.single_action_space, gym.spaces.Box):
        envs = gym.wrappers.vector.ClipAction(envs)
    if obs_norm:
        envs = gym.wrappers.vector.NormalizeObservation(envs)
        envs = gym.wrappers.vector.TransformObservation(
            envs, lambda obs: np.clip(obs, -10.0, 10.0)
        )
        if test_obs_rms is not None:
            envs.env._update_running_mean = False
            envs.env.obs_rms = test_obs_rms
    return envs


class RunningMeanStd:
    """Running mean/std (tianshou-style)."""

    def __init__(
        self,
        mean: float | np.ndarray = 0.0,
        std: float | np.ndarray = 1.0,
        clip_max: float | None = 10.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:
        self.mean, self.var = mean, std
        self.clip_max = clip_max
        self.count = 0
        self.eps = epsilon

    def norm(self, data_array):
        data_array = (data_array - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_max:
            data_array = np.clip(data_array, -self.clip_max, self.clip_max)
        return data_array

    def update(self, data_array: np.ndarray) -> None:
        batch_mean = np.mean(data_array, axis=0)
        batch_var = np.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count


def _build_mlp(
    input_dim: int, output_dim: int, hidden_sizes: list[int], activation: nn.Module
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: list[int], activation: nn.Module):
        super().__init__()
        self.network = _build_mlp(obs_dim, 1, hidden_sizes, activation)

    def forward(self, obs):
        return self.network(obs)


class SumAdvantageNetwork(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int, hidden_sizes: list[int], activation: nn.Module
    ):
        super().__init__()
        self.network = _build_mlp(obs_dim + act_dim, 1, hidden_sizes, activation)

    def forward(self, obs, action):
        return self.network(torch.cat([obs, action], dim=-1))


class Agent(nn.Module):
    """PDA for true discrete action spaces (matches ``pdarl.agent_pda.discrete.PDA_DSC``)."""

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        args: Args,
        device: torch.device,
    ):
        super().__init__()
        self.args = args
        self.device = device

        obs_space = envs.single_observation_space
        act_space = envs.single_action_space
        assert isinstance(act_space, gym.spaces.Discrete), (
            "PDA_DSC only supports discrete action spaces"
        )

        self.obs_dim = int(np.array(obs_space.shape).prod())
        self.act_dim = int(act_space.n)
        self.num_act = self.act_dim

        self.beta = 1
        self.sum_beta = 1
        self.reg = args.step_size
        self.act_noise = args.act_noise
        self.adv_norm = args.adv_norm
        self.ret_norm = args.ret_norm
        self.recompute_ret = args.recompute_ret
        self._eps = 1e-8
        self.act_prob: torch.Tensor | None = None

        if self.ret_norm:
            self.ret_rms = RunningMeanStd()

        activation = getattr(nn, args.activation)()
        self.V = ValueNetwork(self.obs_dim, args.hidden_sizes, activation)
        # Mirror ``PDA_BASE`` then ``PDA_DSC``: a discarded full-action SumAdv is
        # constructed first so PyTorch init RNG matches ``python -m pdarl``.
        _ = SumAdvantageNetwork(
            self.obs_dim, self.act_dim, args.hidden_sizes, activation
        )
        self.SumAdv = SumAdvantageNetwork(
            self.obs_dim, 1, args.hidden_sizes, nn.Tanh()
        )
        self.act = (
            torch.arange(self.num_act, device=self.device).unsqueeze(1).float()
        )

        self.optimV = SOAP(self.V.parameters(), lr=args.learning_rate)
        self.optimSumAdv = SOAP(self.SumAdv.parameters(), lr=args.learning_rate)

    def _obs_to_tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            return torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return obs.to(self.device, dtype=torch.float32)

    def compute_act(self, obs) -> torch.Tensor:
        return self.compute_act_direct(self._obs_to_tensor(obs))

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
        return prob.argmax(dim=1).long()

    @torch.no_grad()
    def exploration_noise(self, act: torch.Tensor) -> torch.Tensor:
        del act
        prob = torch.as_tensor(self.act_prob, device=self.device)
        smoothing_factor = self.act_noise / (self.beta**0.3)
        prob.mul_(1 - smoothing_factor).add_(smoothing_factor / self.num_act)
        return torch.multinomial(prob, 1).squeeze(1).long()

    def action_to_env(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim > 1:
            action = torch.argmax(action, dim=-1)
        return action.long().view(-1)

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
            scale = torch.sqrt(
                torch.tensor(
                    self.ret_rms.var + self._eps, device=v_s.device, dtype=v_s.dtype
                )
            )
            v_s = v_s * scale
            v_s_next = v_s_next * scale

        v_s_next = v_s_next * (1.0 - terminated)

        end_flag = dones.clone()
        end_flag[-1] = 1.0

        adv = self.compute_gae(
            v_s, v_s_next, rewards, end_flag, self.args.gamma, self.args.gae_lambda
        )
        unnorm_ret = adv + v_s

        if self.ret_norm:
            ret = unnorm_ret / scale
            valid = (1.0 - init).reshape(-1).bool()
            self.ret_rms.update(unnorm_ret.reshape(-1)[valid].cpu().numpy())
        else:
            ret = unnorm_ret

        return adv, ret

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
                idx = indices[start:end]

                value = self.V(batch_obs[idx]).flatten()
                loss = (batch_returns[idx] - value).pow(2).mean()

                self.optimV.zero_grad()
                loss.backward()
                if maxgrad:
                    nn.utils.clip_grad_norm_(self.V.parameters(), maxgrad)
                self.optimV.step()
                losses.append(loss.detach().item())
        return losses

    def optimize_sumAdv(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_adv: torch.Tensor,
        repeat: int,
        batch_size: int,
        maxgrad: float | None,
    ) -> list[float]:
        losses: list[float] = []
        total_size = batch_obs.shape[0]
        split_batch_size = batch_size if batch_size else total_size

        if batch_actions.dim() == 1:
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

                current_sumadv = self.SumAdv(
                    batch_obs[idx], batch_actions[idx]
                ).flatten()
                target = (
                    previous_sumadv[idx] * (1.0 - self.beta / self.sum_beta)
                    + batch_adv_norm[idx] * self.beta / self.sum_beta
                )
                loss = (current_sumadv - target).pow(2).mean()

                self.optimSumAdv.zero_grad()
                loss.backward()
                if maxgrad:
                    nn.utils.clip_grad_norm_(self.SumAdv.parameters(), maxgrad)
                self.optimSumAdv.step()
                losses.append(loss.detach().item())
        return losses

    def update_policy(
        self, rollout_data: dict, iteration: int, num_iterations: int
    ) -> dict:
        obs = rollout_data["obs"]
        actions = rollout_data["actions"]
        init = rollout_data["init"]

        if self.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * self.args.learning_rate
            for opt in (self.optimV, self.optimSumAdv):
                opt.param_groups[0]["lr"] = lrnow

        if self.recompute_ret:
            self._rollout_data = rollout_data.copy()

        advantages, returns = self.compute_returns(rollout_data)

        b_valid = (1.0 - init).reshape(-1).bool()
        self._b_valid = b_valid

        b_obs = obs.reshape((-1,) + obs.shape[2:])[b_valid]
        b_actions = actions.reshape((-1,) + actions.shape[2:])[b_valid]
        b_advantages = advantages.reshape(-1)[b_valid]
        b_returns = returns.reshape(-1)[b_valid]

        v_losses = self.optimize_V(
            b_obs,
            b_returns,
            self.args.update_epochs,
            self.args.minibatch_size,
            self.args.max_grad_norm,
        )

        if self.recompute_ret:
            advantages, returns = self.compute_returns(rollout_data)
            b_advantages = advantages.reshape(-1)[b_valid]
            b_returns = returns.reshape(-1)[b_valid]

        sumadv_losses = self.optimize_sumAdv(
            b_obs,
            b_actions,
            b_advantages,
            self.args.update_epochs,
            self.args.minibatch_size,
            self.args.max_grad_norm,
        )

        self.beta += 1
        self.sum_beta += self.beta

        with torch.no_grad():
            y_pred = self.V(b_obs).reshape(-1).cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "sumadv_loss": float(np.mean(sumadv_losses)) if sumadv_losses else 0.0,
            "actor_loss": -1.0,
            "explained_var": explained_var,
            "learning_rate": self.optimV.param_groups[0]["lr"],
            "beta": self.beta,
        }


def evaluate(
    agent: Agent,
    test_envs: gym.vector.VectorEnv,
    device: torch.device,
    num_episodes: int,
) -> dict:
    was_training = agent.training
    agent.eval()
    try:
        next_obs, _ = test_envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)

        eps_returns: list[float] = []
        episode_count = 0

        while episode_count < num_episodes:
            with torch.no_grad():
                action = agent.compute_act(next_obs).detach()
            env_action = agent.action_to_env(action)
            next_obs_np, _, terminations, truncations, infos = test_envs.step(
                env_action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            next_obs = torch.Tensor(next_obs_np).to(device)

            if "episode" in infos and "_episode" in infos:
                done_envs = infos["_episode"]
                if int(done_envs.sum()) > 0:
                    eps_returns.extend(infos["episode"]["r"][done_envs].tolist())
                    episode_count += int(next_done.sum())
            else:
                episode_count += int(next_done.sum())
    finally:
        if was_training:
            agent.train()

    return {
        "test_mean_return": float(np.mean(eps_returns)) if eps_returns else 0.0,
        "test_std_return": float(np.std(eps_returns)) if eps_returns else 0.0,
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = None
    if args.logging:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.deterministic = args.torch_deterministic

    rng = np.random.default_rng(seed=args.seed)
    seeds_train = [
        rng.integers(0, 2**31, dtype=np.int32) for _ in range(args.num_envs)
    ]
    seeds_test = [
        rng.integers(0, 2**31, dtype=np.int32) for _ in range(args.num_test_envs)
    ]

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    envs = make_vector_env(
        env_id=args.env_id,
        num_envs=args.num_envs,
        seeds=seeds_train,
        obs_norm=args.obs_norm,
        test_obs_rms=None,
        env_kargs=args.env_kargs,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    initial_train_obs, _ = envs.reset()

    test_envs = None
    if args.num_test_envs > 0:
        shared_obs_rms = envs.env.obs_rms if args.obs_norm else None
        test_envs = make_vector_env(
            env_id=args.env_id,
            num_envs=args.num_test_envs,
            seeds=seeds_test,
            obs_norm=args.obs_norm,
            test_obs_rms=shared_obs_rms,
            env_kargs=args.env_kargs,
        )
        test_envs.reset()

    agent = Agent(envs, args, device).to(device)

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    terminated = torch.zeros((args.num_steps, args.num_envs)).to(device)
    init = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(initial_train_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        eps_returns: list[float] = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            init[step] = next_done

            with torch.no_grad():
                action = agent.compute_act(next_obs).detach()
                action = agent.exploration_noise(action)
            actions[step] = action

            env_action = agent.action_to_env(action)
            next_obs_np, reward, terminations, truncations, infos = envs.step(
                env_action.cpu().numpy()
            )
            next_done_np = np.logical_or(terminations, truncations)
            terminated[step] = torch.Tensor(terminations).to(device)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)
            dones[step] = next_done

            if "episode" in infos:
                done_envs = infos["_episode"]
                if int(done_envs.sum()) > 0:
                    eps_returns.append(infos["episode"]["r"][done_envs].mean().item())

        if eps_returns and writer is not None:
            writer.add_scalar(
                "charts/episodic_return", float(np.mean(eps_returns)), global_step
            )

        rollout_data = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "init": init,
            "terminated": terminated,
            "next_obs": next_obs,
            "next_done": next_done,
        }
        stats = agent.update_policy(rollout_data, iteration, args.num_iterations)

        if test_envs is not None and iteration % args.test_interval == 0:
            test_stats = evaluate(
                agent, test_envs, device, num_episodes=args.num_test_envs
            )
            print(
                f"global_step={global_step}, "
                f"test_mean_return={test_stats['test_mean_return']:.2f} "
                f"(std={test_stats['test_std_return']:.2f})"
            )
            if writer is not None:
                writer.add_scalar(
                    "charts/test_mean_return",
                    test_stats["test_mean_return"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/test_std_return", test_stats["test_std_return"], global_step
                )

        if writer is not None:
            writer.add_scalar(
                "charts/learning_rate", stats["learning_rate"], global_step
            )
            writer.add_scalar("losses/value_loss", stats["v_loss"], global_step)
            writer.add_scalar("losses/sumadv_loss", stats["sumadv_loss"], global_step)
            writer.add_scalar("losses/actor_loss", stats["actor_loss"], global_step)
            writer.add_scalar(
                "losses/explained_variance", stats["explained_var"], global_step
            )
            writer.add_scalar("charts/beta", stats["beta"], global_step)
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
        # print("SPS:", int(global_step / (time.time() - start_time)))

    if args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if test_envs is not None:
        test_envs.close()
    if writer is not None:
        writer.close()
