class DDPGRunningStatistic:

    epsiode: int
    frame: int
    avg_main_rollout_reward: float
    avg_n_rollout_reward: float
    avg_combined_reward: float
    is_demonstration: bool
    stddev: float
    epsilon: float
    avg_batch_kl_divergence: float
    distance_threshold: float
    critic_loss: float
    actor_loss: float

    def __init__(
            self,
            epsiode: int,
            frame: int,
            avg_main_rollout_reward: float,
            avg_n_rollout_reward: float,
            is_demonstration: bool,
            stddev: float,
            epsilon: float,
            avg_batch_kl_divergence: float,
            distance_threshold: float,
            critic_loss: float,
            actor_loss: float
        ) -> None:
        self.epsiode = epsiode
        self.frame = frame
        self.avg_main_rollout_reward = avg_main_rollout_reward
        self.avg_n_rollout_reward = avg_n_rollout_reward
        self.is_demonstration = is_demonstration
        self.stddev = stddev
        self.epsilon = epsilon
        self.avg_batch_kl_divergence = avg_batch_kl_divergence
        self.distance_threshold = distance_threshold
        self.critic_loss = critic_loss
        self.actor_loss = actor_loss

        self.avg_combined_reward = (avg_main_rollout_reward + avg_n_rollout_reward) / 2.0
    