class RLHyperparametersModel:
    gamma: float  # Discount factor for past rewards
    learning_rate: float
    batch_size: int # Size of batch taken from replay buffer
    max_steps_per_episode: int
    training_episodes: int
    test_episodes: int
    feature_count: int
    action_count: int
    
    episodes: int

    def __init__(
            self,
            gamma: float,
            learning_rate: float,
            batch_size: int,
            training_episodes: int,
            test_episodes: int,
            max_steps_per_episode: int,
            feature_count: int,
            action_count: int
        ):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_episodes = training_episodes
        self.test_episodes = test_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.feature_count = feature_count
        self.action_count = action_count

        self.episodes = training_episodes + test_episodes
