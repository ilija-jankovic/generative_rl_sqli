class DDPGRunningStatistic:

    epsiode: int
    frame: int
    total_avg_reward: float
    is_demonstration: bool
    epsilon: float

    def __init__(
            self,
            epsiode: int,
            frame: int,
            total_avg_reward: float,
            is_demonstration: bool,
            epsilon: float,
        ) -> None:
        self.epsiode = epsiode
        self.frame = frame
        self.total_avg_reward = total_avg_reward
        self.is_demonstration = is_demonstration
        self.epsilon = epsilon