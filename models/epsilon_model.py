class EpsilonModel:
    epsilon: float  # Epsilon greedy parameter
    epsilon_min: float # Minimum epsilon greedy parameter
    epsilon_max: float # Maximum epsilon greedy parameter
    epsilon_interval: tuple  # Rate at which to reduce chance of random action being taken

    def __init__(self, start: float, min: float, max: float):
        self.epsilon = start
        self.epsilon_min = min
        self.epsilon_max = max
        self.epsilon_interval = (
            max - min
        )