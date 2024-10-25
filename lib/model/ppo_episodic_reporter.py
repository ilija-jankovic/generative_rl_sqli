from typing import List
from .running_episodic_statistics import RunningEpisodicStatistics
from .total_episodic_statistics import TotalEpisodicStatistics
from .ppo_reporter import PPOReporter


class PPOEpisodicReporter:
    batch_size: int
    reporter: PPOReporter
    
    __all_running_stats: List[RunningEpisodicStatistics]


    def __init__(
        self,
        batch_size: int,
        reporter: PPOReporter,
    ):
        self.batch_size = batch_size
        self.reporter = reporter

        self.__all_running_stats = [
            RunningEpisodicStatistics(
                episode=1,
                batch_size=batch_size
            ),
        ]

    
    def record_episodic_statistics(
        self,
        episode: int,
        reward: float,
        batch_index: int,
    ):
        assert(batch_index >= 0)
        assert(batch_index < self.batch_size)
        assert(episode >= 1)
        
        base_running_stats = self.__all_running_stats[0]

        assert(episode >= base_running_stats.episode)
        
        latest_recorded_episode = self.__all_running_stats[-1].episode

        if episode > latest_recorded_episode:
            assert(episode - latest_recorded_episode == 1)
            
            self.__all_running_stats.append(
                RunningEpisodicStatistics(
                    episode=episode,
                    batch_size=self.batch_size,
                )
            )

        running_stats = next(
            stats for stats in self.__all_running_stats \
            if stats.episode == episode
        )
        
        running_stats.record_reward(
            reward=reward,
            batch_index=batch_index,
        )

        # If the base episode is still running, it should still be tracked.
        #
        # If the episode is above the base, but is not populated with reward,
        # the base episode is still running.
        #
        # If the current episode is not the base episode, and is populated
        # with reward, it must be one episode after the base. In this case,
        # the base episode should be completed and removed.
        if base_running_stats == running_stats or not running_stats.is_populated:
            return
        
        total_stats = TotalEpisodicStatistics(
            episode=base_running_stats.episode,
            mean_cumulative_episodic_reward=base_running_stats.mean_running_reward,
            mean_accuracy=base_running_stats.mean_accuracy,
        )

        self.reporter.record_episodic_statistics(total_stats)
        
        print(f'Mean Cumulative Reward: {total_stats.mean_cumulative_episodic_reward}, '
              + f'Mean Accuracy {total_stats.mean_accuracy * 100.0}%')

        # Delete completed batch episode.
        del self.__all_running_stats[0]
