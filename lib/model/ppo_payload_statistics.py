from lib.model.payload import Payload


class PPOPayloadStatistics:
 
    timestep: int
    reward: float
    payload: Payload

    def __init__(
        self,
        timestep: int,
        reward: float,
        payload: Payload,
    ):
        self.timestep = timestep
        self.reward = reward
        self.payload = payload
    