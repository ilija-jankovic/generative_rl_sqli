class DDPGPayloadStatistic:
    
    epsiode: int
    frame: int
    payload: str
    reward: float
    is_demonstration: bool

    def __init__(
            self,
            epsiode: int,
            frame: int,
            payload: str,
            reward: float,
            is_demonstration: bool,
        ) -> None:
        self.epsiode = epsiode
        self.frame = frame
        self.payload = payload
        self.reward = reward
        self.is_demonstration = is_demonstration
