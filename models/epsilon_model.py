class EpsilonModel:
    epsilon: float  # Epsilon greedy parameter
    epsilon_min: float # Minimum epsilon greedy parameter
    epsilon_max: float # Maximum epsilon greedy parameter
    random_frame_count: int # Number of frames to choose a random action
    greedy_frame_count: int # Decay time of epsilon from max to min.
    epsilon_interval: tuple  # Rate at which to reduce chance of random action being taken

    def __init__(
            self,
            start: float,
            min: float,
            max: float,
            random_frame_count: int,
            greedy_frame_count: int):
        self.epsilon = start
        self.epsilon_min = min
        self.epsilon_max = max
        self.random_frame_count = random_frame_count
        self.greedy_frame_count = greedy_frame_count
        self.epsilon_interval = (
            max - min
        )