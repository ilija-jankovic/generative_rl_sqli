from enum import Enum


class PolicyType(Enum):
    NORMAL = 1

    # For DDPG only.
    PERTURBED = 2
    
    # For DDPG only.
    TARGET = 3

    # For PPO only.
    OLD = 4

    # For PPO only.
    SUCCESSFUL_DEMONSTRATIONS = 5

    # For PPO only.
    UNSUCCESSFUL_DEMONSTRATIONS = 6
