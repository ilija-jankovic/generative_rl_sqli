class DDPGRunningStatistic:

    epsiode: int
    frame: int
    total_avg_reward: float
    is_demonstration: bool
    stddev: float
    epsilon: float
    avg_kl_divergence: float

    def __init__(
            self,
            epsiode: int,
            frame: int,
            total_avg_reward: float,
            is_demonstration: bool,
            stddev: float,
            epsilon: float,
            avg_kl_divergence: float
        ) -> None:
        self.epsiode = epsiode
        self.frame = frame
        self.total_avg_reward = total_avg_reward
        self.is_demonstration = is_demonstration
        self.stddev = stddev
        self.epsilon = epsilon
        self.avg_kl_divergence = avg_kl_divergence