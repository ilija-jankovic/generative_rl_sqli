class DDPGRunningStatistic:

    epsiode: int
    frame: int
    total_avg_reward: float
    is_demonstration: bool
    stddev: float
    epsilon: float
    total_avg_kl_divergence: float
    distance_threshold: float

    def __init__(
            self,
            epsiode: int,
            frame: int,
            total_avg_reward: float,
            is_demonstration: bool,
            stddev: float,
            epsilon: float,
            total_avg_kl_divergence: float,
            distance_threshold: float
        ) -> None:
        self.epsiode = epsiode
        self.frame = frame
        self.total_avg_reward = total_avg_reward
        self.is_demonstration = is_demonstration
        self.stddev = stddev
        self.epsilon = epsilon
        self.total_avg_kl_divergence = total_avg_kl_divergence
        self.distance_threshold = distance_threshold