from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class Args:
    exp_name: str = "pda_continuous_action"
    """the name of the experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cpu"
    """if toggled, cuda will be enabled by default"""
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

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    env_kargs: dict = None
    """the arguments for the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_test_envs: int = 10
    """the number of parallel test environments"""
    test_interval: int = 5
    """test the agent every N iterations"""
    num_steps: int = 400
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    """hidden layer sizes for value/sum-advantage/actor MLPs"""
    activation: str = "Tanh"
    """activation function for value/sum-advantage/actor MLPs"""
    obs_norm: bool = True
    """whether to normalize observations"""
    max_grad_norm: float = None
    """the maximum norm for the gradient clipping"""
    ret_norm: bool = True
    """whether to normalize returns"""
    adv_norm: bool = True
    """whether to normalize advantages"""
    recompute_ret: bool = True
    """whether to recompute returns during optimization"""

    # PDA-specific arguments
    agent_name: str = "PDA_ACT"
    """agent implementation to use: PDA_ACT, PDA_OPT, PDA_DSC, PDA_BCD, PMD_ACT"""
    step_size: float = 0.5
    """regularization coefficient (step size) for PDA"""
    act_noise: float = 1.3
    """action noise for exploration"""
    action_opt: str = "RGD"
    """action optimizer for PDA_OPT: RGD or ACFGM"""
    action_opt_itr: int = 100
    """number of optimization iterations for PDA_OPT/PDA_BCD"""
    action_opt_params: list[float] | float | None = None
    """optimizer-specific params for PDA_OPT"""
    discretization: int | None = None
    """number of action bins for PDA_BCD"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def read_yaml_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_args_from_yaml(
    yaml_path: Optional[str] = None,
    default_yaml_path: Optional[str] = None,
    strict: bool = True,
) -> Args:
    """
    Load Args from YAML with optional default + override merge.

    Args:
        yaml_path: Primary YAML path. If omitted, only ``default_yaml_path``
            (when provided and present) or dataclass defaults are used.
        default_yaml_path: Optional base YAML merged before ``yaml_path``.
        strict: If True, raise ValueError for unknown config keys.

    Returns:
        Args instance populated from YAML file
    """
    args = Args()
    known_keys = {f.name for f in fields(Args)}

    config: dict[str, Any] = {}
    if default_yaml_path is not None:
        default_path = Path(default_yaml_path)
        if default_path.exists():
            config.update(read_yaml_config(default_path))

    if yaml_path is not None:
        override_path = Path(yaml_path)
        if not override_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {override_path}. "
                "Download example configs from "
                "https://github.com/JGIoA/pdarl/tree/main/config "
                "or pass a path to your own YAML file."
            )
        config.update(read_yaml_config(override_path))

    unknown_keys = sorted(set(config) - known_keys)
    if strict and unknown_keys:
        raise ValueError(
            "Unknown config keys in YAML: "
            + ", ".join(unknown_keys)
            + ". These keys are not defined in Args."
        )

    for key, value in config.items():
        if key in known_keys:
            setattr(args, key, value)

    return args
