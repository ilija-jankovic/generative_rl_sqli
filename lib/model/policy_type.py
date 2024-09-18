from enum import Enum


class PolicyType(Enum):
    NORMAL = 1

    # For PPO only.
    OLD = 2
