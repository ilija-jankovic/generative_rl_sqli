class PPOPayloadStatistics:
 
    timestep: int
    reward: float
    payload: str

    def __init__(
        self,
        timestep: int,
        reward: float,
        payload: str,
    ):
        self.timestep = timestep
        self.reward = reward
        self.payload = payload
    