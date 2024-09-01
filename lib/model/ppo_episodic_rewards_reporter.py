from typing import List
from .ppo_episodic_statistics import PPOEpisodicStatistics
from .ppo_reporter import PPOReporter

class PPOEpisodicRewardsReporter:
    batch_size: int
    reporter: PPOReporter
    
    __total_episodic_rewards: dict[int, List[float]]
    
    def __init__(
        self,
        batch_size: int,
        reporter: PPOReporter,
    ):
        self.batch_size = batch_size
        self.reporter = reporter

        self.__total_episodic_rewards = {}
    
    def record_reward(self, reward: float, batch_index: int, episode: int):
        assert(batch_index >= 0)
        assert(batch_index < self.batch_size)

        assert(episode >= 1)

        # Define uninitialised rewards list if episodic rewards not yet initialised.
        if episode not in self.__total_episodic_rewards:
            self.__total_episodic_rewards[episode] = [None] * self.batch_size

        if self.__total_episodic_rewards[episode][batch_index] is None:
            self.__total_episodic_rewards[episode][batch_index] = reward
            
            is_new_episodic_reward = True
        else:
            self.__total_episodic_rewards[episode][batch_index] += reward
            
            is_new_episodic_reward = False

        if not is_new_episodic_reward:
            return

        previous_episodic_rewards = self.__total_episodic_rewards.get(episode - 1, [None])

        if None in previous_episodic_rewards:
            return

        assert(None not in previous_episodic_rewards)

        mean_cumulative_episodic_reward = sum(previous_episodic_rewards) / self.batch_size
        
        stats = PPOEpisodicStatistics(
            episode=episode - 1,
            mean_cumulative_episodic_reward=mean_cumulative_episodic_reward,
        )

        self.reporter.record_episodic_statistics(stats)
        
        print(mean_cumulative_episodic_reward)

        # Delete recorded epsiodic reward data.
        del self.__total_episodic_rewards[episode - 1]
        