import random
import time

import gymnasium as gym
import numpy as np
import torch

from .agent_pda import (
    PDA_ACT,
    PDA_BCD,
    PDA_DSC,
    PDA_OPT,
    PMD_ACT,
)
from .utils.args import Args
from .env import Environment


class Trainer:
    """On-policy training and evaluation for PDA agents.

    Typical usage::

        trainer = Trainer(args)
        trainer.setup()
        trainer.run()
    """

    _AGENT_REGISTRY = {
        "PDA_ACT": PDA_ACT,
        "PDA_OPT": PDA_OPT,
        "PDA_DSC": PDA_DSC,
        "PDA_BCD": PDA_BCD,
        "PMD_ACT": PMD_ACT,
    }

    def __init__(self, args=None):
        """
        Initialize the Trainer with configuration arguments.

        Args:
            args: Args instance. If None, creates a new Args with default values.
        """
        self.args = args if args is not None else Args()
        self._setup_args()
        self.run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        self.device = None
        self.env = None
        self.test_env = None
        self.agent = None
        self.start_time = None
        self.action_space = None
        self.writer = None

    def _setup_args(self):
        """Compute and set runtime arguments."""
        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(
            self.args.batch_size // self.args.num_minibatches
        )
        self.args.num_iterations = self.args.total_timesteps // self.args.batch_size

        # Set action limits from environment (will be set after env creation)
        self.args.act_lims = None

    def _setup_seeding(self):
        """Set random seeds for reproducibility."""
        seed = self.args.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.args.device == "cuda":
            torch.backends.cudnn.deterministic = self.args.torch_deterministic

        rng = np.random.default_rng(seed=seed)
        self.seeds_train = [
            rng.integers(0, 2**31, dtype=np.int32) for _ in range(self.args.num_envs)
        ]
        self.seeds_test = [
            rng.integers(0, 2**31, dtype=np.int32)
            for _ in range(self.args.num_test_envs)
        ]

    def _setup_device(self):
        """Setup compute device."""
        if self.args.device == "cuda":
            self.device = torch.device("cuda")
        elif self.args.device == "mps":
            self.device = torch.device("mps")
        elif self.args.device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Invalid device: {self.args.device}")

    def _setup_environment(self):
        """Create and initialize the training and test environments."""
        # Setup training environment
        if self.args.num_envs > 0:
            self.env = Environment(
                env_id=self.args.env_id,
                num_envs=self.args.num_envs,
                obs_norm=self.args.obs_norm,
                test_obs_rms=None,
                seed=self.args.seed,
                seeds=self.seeds_train,
                device=self.device,
                num_steps=self.args.num_steps,
                exploration_noise=True,
                env_kargs=self.args.env_kargs,
            )
            self.action_space = self.env.env.action_space

            # Set action limits from environment
            if isinstance(self.env.env.action_space, gym.spaces.Box):
                self.args.act_lims = [
                    float(self.env.env.action_space.low[0]),
                    float(self.env.env.action_space.high[0]),
                ]

        else:
            self.env = None

        # Setup test environment
        if self.args.num_test_envs > 0:
            test_obs_rms = None
            if self.args.obs_norm and self.env is not None:
                test_obs_rms = self.env.envs.env.obs_rms
            self.test_env = Environment(
                env_id=self.args.env_id,
                num_envs=self.args.num_test_envs,
                obs_norm=self.args.obs_norm,
                test_obs_rms=test_obs_rms,
                seed=self.args.seed,
                seeds=self.seeds_test,
                device=self.device,
                num_steps=self.args.num_steps,
                exploration_noise=False,
                env_kargs=self.args.env_kargs,
            )
            if self.action_space is None:
                self.action_space = self.test_env.env.action_space
            # Set action limits from environment (if not already set)
            if (
                self.args.act_lims is None
                and isinstance(self.test_env.env.action_space, gym.spaces.Box)
            ):
                self.args.act_lims = [
                    float(self.test_env.env.action_space.low[0]),
                    float(self.test_env.env.action_space.high[0]),
                ]
        else:
            self.test_env = None

    def _setup_agent(self):
        """Create and initialize the agent."""
        if self.env is not None:
            obs_space = self.env.env.observation_space
            act_space = self.env.env.action_space
        elif self.test_env is not None:
            obs_space = self.test_env.env.observation_space
            act_space = self.test_env.env.action_space
        else:
            raise ValueError("Either env or test_env must be set")

        obs_dim = int(np.array(obs_space.shape).prod())
        if isinstance(act_space, gym.spaces.Box):
            act_dim = int(np.prod(act_space.shape))
        elif isinstance(act_space, gym.spaces.Discrete):
            act_dim = int(act_space.n)
        else:
            raise NotImplementedError(
                f"Unsupported action space type: {type(act_space).__name__}. "
                "Supported types are Box and Discrete."
            )

        agent_name = getattr(self.args, "agent_name", "PDA_ACT")
        agent_cls = self._AGENT_REGISTRY.get(agent_name)
        if agent_cls is None:
            valid = ", ".join(sorted(self._AGENT_REGISTRY))
            raise ValueError(f"Unknown agent_name: {agent_name}. Valid values: {valid}")
        self.agent = agent_cls(obs_dim, act_dim, self.args, self.device).to(self.device)

    def _to_env_action(self, action: torch.Tensor) -> torch.Tensor:
        """Convert agent-space actions to environment-space actions."""
        if self.action_space is None:
            raise RuntimeError("Action space is not initialized. Call setup() first.")
        return Environment._to_env_action(self.agent, action, self.action_space)

    def _setup_logging(self):
        if self.args.track:
            import wandb

            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=vars(self.args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )

        if self.args.logging:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f"runs/{self.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
            )

    def setup(self):
        """Complete setup: seeding, device, environment, agent, and logging."""
        self._setup_seeding()
        self._setup_device()
        self._setup_environment()
        self._setup_agent()
        self._setup_logging()
        self.start_time = time.time()

    def evaluate(self, num_episodes: int) -> dict:
        """
        Evaluate the agent on test environments without exploration noise.

        Args:
            num_episodes: Number of episodes to evaluate (across all test envs)

        Returns:
            Dictionary containing evaluation statistics:
            - test_mean_return: mean episodic return
            - test_std_return: standard deviation of mean episodic returns
        """
        if self.test_env is None or self.agent is None:
            raise RuntimeError(
                "Test environment and agent must be set up before evaluation. Call setup() first."
            )

        # Reset test environments
        test_obs = self.test_env.reset()
        test_episodic_returns = []
        episode_count = 0
        was_training = self.agent.training
        self.agent.eval()

        # Run evaluation episodes
        try:
            while episode_count < num_episodes:
                action = self.agent.Actor(test_obs).detach()

                # Execute step
                env_action = self._to_env_action(action)
                next_obs, reward, terminations, truncations, infos = (
                    self.test_env.envs.step(env_action.cpu().numpy())
                )
                next_done = np.logical_or(terminations, truncations)
                test_obs = torch.Tensor(next_obs).to(self.device)

                if "episode" in infos and "_episode" in infos:
                    eps_info = infos["episode"]
                    done_envs = infos["_episode"]
                    if len(done_envs) > 0:
                        eps_ret = eps_info["r"][done_envs]
                        test_episodic_returns.extend(eps_ret.tolist())
                        episode_count += int(next_done.sum())
                else:
                    # Fallback to avoid hanging if episode statistics keys are absent.
                    episode_count += int(next_done.sum())

        finally:
            if was_training:
                self.agent.train()

        # Compute statistics
        test_mean_return = (
            np.mean(test_episodic_returns) if test_episodic_returns else 0.0
        )
        test_std_return = (
            np.std(test_episodic_returns) if test_episodic_returns else 0.0
        )

        return {
            "test_mean_return": test_mean_return,
            "test_std_return": test_std_return,
        }

    def train(self):
        """Run the training loop."""
        if self.env is None or self.agent is None:
            raise RuntimeError(
                "Trainer must be set up before training. Call setup() first."
            )

        for iteration in range(1, self.args.num_iterations + 1):
            rollout_data = self.env.collect_rollout(self.agent, self.args.num_steps)

            stats = self.agent.update_policy(
                rollout_data, iteration, self.args.num_iterations
            )

            if iteration % self.args.test_interval == 0:
                test_stats = self.evaluate(num_episodes=self.args.num_test_envs)
                print(
                    f"global_step={self.env.global_step}, test_mean_return={test_stats['test_mean_return']:.2f} "
                    f"(std={test_stats['test_std_return']:.2f})"
                )
                if self.writer is not None:
                    self.writer.add_scalar("charts/test_mean_return", test_stats["test_mean_return"], self.env.global_step)
                    self.writer.add_scalar("charts/test_std_return", test_stats["test_std_return"], self.env.global_step)

            if self.writer is not None:
                self.writer.add_scalar("charts/learning_rate", stats["learning_rate"], self.env.global_step)
                self.writer.add_scalar("losses/value_loss", stats["v_loss"], self.env.global_step)
                self.writer.add_scalar("losses/sumadv_loss", stats["sumadv_loss"], self.env.global_step)
                self.writer.add_scalar("losses/actor_loss", stats["actor_loss"], self.env.global_step)
                self.writer.add_scalar("losses/explained_variance", stats["explained_var"], self.env.global_step)
                self.writer.add_scalar("charts/beta", stats["beta"], self.env.global_step)
                self.writer.add_scalar("charts/SPS", int(self.env.global_step / (time.time() - self.start_time)), self.env.global_step)

    def cleanup(self):
        """Clean up resources (close environment, writer, etc.)."""
        if self.env is not None:
            self.env.close()
        if self.test_env is not None:
            self.test_env.close()
        if self.writer is not None:
            self.writer.close()

    def run(self):
        """Complete training pipeline: train, test, and cleanup."""
        try:
            self.train()
        finally:
            self.cleanup()
