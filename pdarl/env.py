import gymnasium as gym
import numpy as np
import torch


class Environment:
    def __init__(
        self,
        env_id: str,
        num_envs: int,
        obs_norm: bool,
        test_obs_rms: object,
        seed: int,
        seeds: list[int],
        device: str,
        num_steps: int,
        exploration_noise: bool,
        env_kargs: dict | None = None,
    ):
        self.env_id = env_id
        self.num_envs = num_envs
        self.device = device
        self.num_steps = num_steps
        self.global_step = 0
        self.exploration_noise = exploration_noise
        self.env_kargs = env_kargs or {}

        self.env = gym.make(env_id, **self.env_kargs)
        # Create vectorized environments
        envs = gym.vector.SyncVectorEnv(
            [
                (
                    lambda seed=seeds[i]: self.create_env(
                        self.env_id, seed=seed, env_kargs=self.env_kargs
                    )
                )
                for i in range(num_envs)
            ],
        )

        envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
        if isinstance(self.env.action_space, gym.spaces.Box):
            envs = gym.wrappers.vector.ClipAction(envs)

        if obs_norm:
            # Normalize observations and clip to [-10, 10]
            # Consistent with Tianshou implementation
            envs = gym.wrappers.vector.NormalizeObservation(envs)
            envs = gym.wrappers.vector.TransformObservation(
                envs, lambda obs: np.clip(obs, -10, 10)
            )
            # Disable running mean update for test environments
            # Use training obs_rms for normalization
            if test_obs_rms is not None:
                envs.env._update_running_mean = False
                envs.env.obs_rms = test_obs_rms

        self.envs = envs

        # Initialize rollout storage tensors
        self.obs = torch.zeros(
            (num_steps, num_envs) + self.envs.single_observation_space.shape
        ).to(device)
        self.actions = torch.zeros(
            (num_steps, num_envs) + self.envs.single_action_space.shape
        ).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.terminated = torch.zeros((num_steps, num_envs)).to(device)
        self.init = torch.zeros((num_steps, num_envs)).to(device)

        # Initialize environment state
        self.next_obs, _ = self.envs.reset()
        self.next_obs = torch.Tensor(self.next_obs).to(device)
        self.next_done = torch.zeros(num_envs).to(device)

    def reset(self, seed=None):
        """Reset environments and return initial observations."""
        self.next_obs, _ = self.envs.reset(seed=seed)
        self.next_obs = torch.Tensor(self.next_obs).to(self.device)
        self.next_done = torch.zeros(self.num_envs).to(self.device)
        return self.next_obs

    @staticmethod
    def _to_env_action(
        agent, action: torch.Tensor, action_space: gym.spaces.Space
    ) -> torch.Tensor:
        """Convert agent-space actions to environment-space actions."""
        if hasattr(agent, "action_to_env"):
            action = agent.action_to_env(action)
        elif isinstance(action_space, gym.spaces.Box) and hasattr(
            agent.Actor, "map_action"
        ):
            action = agent.Actor.map_action(action)

        if isinstance(action_space, gym.spaces.Discrete):
            if action.ndim > 1:
                # If a policy outputs logits/probabilities, take greedy action.
                action = torch.argmax(action, dim=-1)
            return action.long().view(-1)

        return action

    def collect_rollout(self, agent, num_steps):
        """
        Collect rollout data using the agent.

        Args:
            agent: Agent instance to use for action selection
            num_steps: Number of steps to collect

        Returns:
            Dictionary containing rollout data with keys:
            - obs: observations
            - actions: actions taken (with exploration noise)
            - rewards: rewards received
            - dones: done flags
            - values: value estimates
            - next_obs: next observations after rollout
            - next_done: next done flags after rollout
            - episodic_returns: list of episodic returns
        """


        eps_returns = []
        for step in range(num_steps):

            self.global_step += self.num_envs
            self.obs[step] = self.next_obs
            self.init[step] = self.next_done

            # Obtain actions from actor
            action = agent.Actor(self.next_obs).detach()
            if self.exploration_noise:
                action = agent.exploration_noise(action)
            self.actions[step] = action

            # Step environment
            env_action = self._to_env_action(agent, action, self.env.action_space)
            next_obs, reward, terminations, truncations, infos = self.envs.step(
                env_action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            self.terminated[step] = torch.Tensor(terminations).to(self.device)
            self.rewards[step] = torch.tensor(reward).to(self.device)
            self.next_obs, self.next_done = (
                torch.Tensor(next_obs).to(self.device),
                torch.Tensor(next_done).to(self.device),
            )

            self.dones[step] = self.next_done
            if "episode" in infos:
                eps_info = infos["episode"]
                done_envs = infos["_episode"]
                eps_ret = eps_info["r"][done_envs].mean().item()
                eps_returns.append(eps_ret)

        return {
            "obs": self.obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "init": self.init,
            "next_obs": self.next_obs,
            "next_done": self.next_done,
            "terminated": self.terminated,
            "episodic_returns": eps_returns,
        }

    def close(self):
        """Close environments."""
        self.envs.close()

    @staticmethod
    def create_env(
        task: str, seed: int | None = None, env_kargs: dict | None = None
    ) -> gym.Env:
        if env_kargs is None:
            env = gym.make(task)
        else:
            env = gym.make(task, **env_kargs)
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
            env.np_random = rng
            if hasattr(env, "_np_random_seed"):
                env._np_random_seed = seed
        return env
