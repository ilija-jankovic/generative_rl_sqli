class DDPGRunningStatistic:

    epsiode: int
    frame: int
    avg_reward: float
    is_demonstration: bool
    adpative_sigma: float
    adpative_delta: float

    def __init__(
            self,
            epsiode: int,
            frame: int,
            avg_reward: float,
            is_demonstration: bool,
            adpative_sigma: float,
            adpative_delta: float,
        ) -> None:
        self.epsiode = epsiode
        self.frame = frame
        self.avg_reward = avg_reward
        self.is_demonstration = is_demonstration
        self.adpative_sigma = adpative_sigma
        self.adpative_delta = adpative_delta