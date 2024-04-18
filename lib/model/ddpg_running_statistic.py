class DDPGRunningStatistic:

    epsiode: int
    frame: int
    avg_main_rollout_reward: float

    # Deprecated. n-Step rollouts removed.
    avg_n_rollout_reward: float

    avg_combined_reward: float
    is_demonstration: bool
    critic_loss: float
    actor_loss: float

    def __init__(
            self,
            epsiode: int,
            frame: int,
            avg_main_rollout_reward: float,
            is_demonstration: bool,
            critic_loss: float,
            actor_loss: float
        ) -> None:
        self.epsiode = epsiode
        self.frame = frame

        self.avg_main_rollout_reward = avg_main_rollout_reward

        # NOTE: Same value as self.avg_main_rollout_reward as
        # n-Step rollouts removed.
        self.avg_n_rollout_reward = avg_main_rollout_reward

        self.is_demonstration = is_demonstration
        self.critic_loss = critic_loss
        self.actor_loss = actor_loss

        self.avg_combined_reward = avg_main_rollout_reward
    