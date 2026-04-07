"""
Phase 4: Experiment configurations

Defines ExperimentConfig (one run specification) and get_all_configs()
which returns the full 90-config factorial experiment matrix.
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import List

VALID_ALGORITHMS = ('mc', 'qlearning', 'expected_sarsa', 'double_q', 'value_iter')
VALID_SETTINGS   = ('high', 'moderate', 'low')

# n_episodes defaults by algorithm
_DEFAULT_N_EPISODES = {
    'mc'            : 5000,
    'value_iter'    : 200,   # evaluation rollouts post-solve
    'qlearning'     : 5000,
    'expected_sarsa': 5000,
    'double_q'      : 5000,
}


@dataclass
class ExperimentConfig:
    """
    Fully specifies one experiment (algorithm × environment × hyperparameters).

    Fields
    ------
    setting        : EEG response separation — 'high' | 'moderate' | 'low'
    horizon        : episode length I ∈ {5, 10}
    c_switch       : site-switching penalty ∈ {0, 0.1, 0.25}
    algorithm      : one of VALID_ALGORITHMS
    alpha          : learning rate (model-free agents only)
    epsilon        : initial exploration probability
    epsilon_decay  : multiplicative ε decay per episode
    epsilon_min    : floor on ε
    n_episodes     : training episodes (TD/MC) or evaluation rollouts (VI)
    n_seeds        : number of independent random seeds to average over
    """
    setting       : str
    horizon       : int
    c_switch      : float
    algorithm     : str
    alpha         : float = 0.1
    epsilon       : float = 1.0
    epsilon_decay : float = 0.995
    epsilon_min   : float = 0.05
    n_episodes    : int   = 5000
    n_seeds       : int   = 10

    def __post_init__(self) -> None:
        if self.setting not in VALID_SETTINGS:
            raise ValueError(f"setting must be one of {VALID_SETTINGS}, got '{self.setting}'")
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {VALID_ALGORITHMS}, got '{self.algorithm}'")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.c_switch < 0:
            raise ValueError(f"c_switch must be >= 0, got {self.c_switch}")
        if self.n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {self.n_episodes}")
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")

    def config_hash(self) -> str:
        """Short (12-char) deterministic hex hash identifying this config."""
        return hashlib.md5(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:12]


def get_all_configs() -> List[ExperimentConfig]:
    """
    Return the full factorial experiment matrix:
      3 settings × 2 horizons × 3 switch costs × 5 algorithms = 90 configs.
    """
    configs = []
    for setting in ('high', 'moderate', 'low'):
        for horizon in (5, 10):
            for c_switch in (0.0, 0.1, 0.25):
                for algorithm in VALID_ALGORITHMS:
                    configs.append(ExperimentConfig(
                        setting   = setting,
                        horizon   = horizon,
                        c_switch  = c_switch,
                        algorithm = algorithm,
                        n_episodes = _DEFAULT_N_EPISODES[algorithm],
                    ))
    return configs
