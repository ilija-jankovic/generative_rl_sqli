from enum import Enum


class PolicyType(Enum):
    NORMAL = 1

    # For PPO only.
    OLD = 2

    # For PPO only.
    SUCCESSFUL_DEMONSTRATIONS = 3
