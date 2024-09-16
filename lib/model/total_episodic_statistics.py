class TotalEpisodicStatistics:
    
    episode: int
    mean_cumulative_episodic_reward: float
    mean_accuracy: float


    def __init__(
        self,
        episode: int,
        mean_cumulative_episodic_reward: float,
        mean_accuracy: float,
    ):
        self.episode = episode
        self.mean_cumulative_episodic_reward = mean_cumulative_episodic_reward
        self.mean_accuracy = mean_accuracy