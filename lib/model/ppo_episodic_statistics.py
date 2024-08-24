class PPOEpisodicStatistics:
    
    episode: int
    mean_cumulative_episodic_reward: float

    def __init__(
        self,
        episode: int,
        mean_cumulative_episodic_reward: float,
    ):
        self.episode = episode
        self.mean_cumulative_episodic_reward = mean_cumulative_episodic_reward