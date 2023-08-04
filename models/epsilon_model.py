class EpsilonModel:
    epsilon: float  # Epsilon greedy parameter
    epsilon_min: float # Minimum epsilon greedy parameter
    epsilon_max: float # Maximum epsilon greedy parameter
    num_random_frames: int # Number of frames to choose a random action
    epsilon_interval: tuple  # Rate at which to reduce chance of random action being taken

    def __init__(
            self,
            start: float,
            min: float,
            max: float,
            num_random_frames: int):
        self.epsilon = start
        self.epsilon_min = min
        self.epsilon_max = max
        self.num_random_frames = num_random_frames
        self.epsilon_interval = (
            max - min
        )