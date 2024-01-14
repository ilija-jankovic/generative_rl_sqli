class DDPGRunningStatistic:

    epsiode: int
    frame: int
    total_avg_reward: float
    is_demonstration: bool
    adpative_sigma: float
    adpative_delta: float
    avg_perturbation_distance: float

    def __init__(
            self,
            epsiode: int,
            frame: int,
            total_avg_reward: float,
            is_demonstration: bool,
            adpative_sigma: float,
            adpative_delta: float,
            avg_perturbation_distance: float,
        ) -> None:
        self.epsiode = epsiode
        self.frame = frame
        self.total_avg_reward = total_avg_reward
        self.is_demonstration = is_demonstration
        self.adpative_sigma = adpative_sigma
        self.adpative_delta = adpative_delta
        self.avg_perturbation_distance = avg_perturbation_distance