import unittest

from lib.model.ppo_episodic_reporter import PPOEpisodicReporter

from .mocks.mock_ppo_reporter import MockPPOReporter


class TestPPOEpisodicReporter(unittest.TestCase):

    __reporter: PPOEpisodicReporter


    def setUp(self) -> None:
        self.__reporter = PPOEpisodicReporter(
            batch_size=4,
            reporter=MockPPOReporter(),
        )
        
        
    def __record_sequential_positive_rewards(self, episode: int):
        self.__reporter.record_episodic_statistics(
            episode=episode,
            reward=1.0,
            batch_index=0,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=episode,
            reward=2.0,
            batch_index=1,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=episode,
            reward=3.0,
            batch_index=2,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=episode,
            reward=4.0,
            batch_index=3,
        )
        
    
    def __get_stats(self):
        reporter: MockPPOReporter = self.__reporter.reporter
        
        return reporter.stats


    def test_populate_episode(self):
        '''
        Populating episode rewards does not record their cumulative
        mean.
        '''

        self.__record_sequential_positive_rewards(episode=1)
        stats = self.__get_stats()

        self.assertListEqual(stats, [])
        

    def test_populate_episode_then_add_to_subsequent(self):
        '''
        Populating episode rewards then recording a subsequent episode
        does not record the initial cumulative episodic reward mean.
        '''

        self.__record_sequential_positive_rewards(episode=1)
        
        self.__reporter.record_episodic_statistics(
            episode=2,
            reward=0.0,
            batch_index=0,
        )
        
        stats = self.__get_stats()

        self.assertListEqual(stats, [])


    def test_populate_episode_then_populate_subsequent(self):
        '''
        Populating episode rewards then populating subsequent episodic
        rewards records the initial cumulative episodic reward mean.
        '''

        self.__record_sequential_positive_rewards(episode=1)
        self.__record_sequential_positive_rewards(episode=2)

        self.__reporter.record_episodic_statistics(
            episode=2,
            reward=99999.0,
            batch_index=0,
        )
        
        stats = self.__get_stats()

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].mean_cumulative_episodic_reward, 2.5)


    def test_populate_two_episodes_then_add_to_three(self):
        '''
        Populating episode rewards, then populating subsequent episode
        rewards records, then adding to subsequent two episodic rewards,
        records the initial cumulative episodic reward mean.
        '''

        self.__record_sequential_positive_rewards(episode=1)
        self.__record_sequential_positive_rewards(episode=2)

        self.__reporter.record_episodic_statistics(
            episode=2,
            reward=10.0,
            batch_index=2,
        )

        self.__reporter.record_episodic_statistics(
            episode=3,
            reward=7.0,
            batch_index=2,
        )

        self.__reporter.record_episodic_statistics(
            episode=3,
            reward=4.0,
            batch_index=3,
        )

        self.__reporter.record_episodic_statistics(
            episode=4,
            reward=4.0,
            batch_index=2,
        )
        
        stats = self.__get_stats()
        
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].mean_cumulative_episodic_reward, 2.5)
        

    def test_populate_three_episodes_then_add_to_three(self):
        '''
        Populating episode rewards, then populating subsequent two episode
        rewards records, then adding to subsequent two episodic rewards,
        records the first two cumulative episodic reward means.
        '''

        self.__record_sequential_positive_rewards(episode=1)
        self.__record_sequential_positive_rewards(episode=2)

        self.__reporter.record_episodic_statistics(
            episode=2,
            reward=10.0,
            batch_index=0,
        )

        self.__record_sequential_positive_rewards(episode=3)

        self.__reporter.record_episodic_statistics(
            episode=4,
            reward=7.0,
            batch_index=0,
        )

        self.__reporter.record_episodic_statistics(
            episode=4,
            reward=56.0,
            batch_index=1,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=4,
            reward=4.0,
            batch_index=0,
        )
        
        stats = self.__get_stats()

        self.assertEqual(len(stats), 2)
        self.assertEqual(stats[0].mean_cumulative_episodic_reward, 2.5)
        self.assertEqual(stats[1].mean_cumulative_episodic_reward, 5)


    def test_full_accuracy_full_positive_reward(self):
        '''
        Populating positive episode rewards then populating subsequent
        episodic rewards records the initial episodic accuary mean as
        100%.
        '''
        
        self.__record_sequential_positive_rewards(episode=1)
        self.__record_sequential_positive_rewards(episode=2)
        
        stats = self.__get_stats()
        
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].mean_accuracy, 1.0)
        

    def test_half_accuracy_half_positive_reward(self):
        '''
        Populating half-positive episode rewards then populating
        subsequent episodic rewards records the initial episodic
        accuary mean as 50%.
        '''
        
        self.__reporter.record_episodic_statistics(
            episode=1,
            reward=0.0,
            batch_index=0,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=1,
            reward=0.0,
            batch_index=1,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=1,
            reward=1.0,
            batch_index=2,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=1,
            reward=1.0,
            batch_index=3,
        )
        
        self.__record_sequential_positive_rewards(episode=2)
        
        stats = self.__get_stats()
        
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].mean_accuracy, 0.5)


    def test_no_accuracy_negative_rewards(self):
        '''
        Populating negative episode rewards then populating
        subsequent episodic rewards records the initial episodic
        accuary mean as 0%.
        '''
        
        self.__reporter.record_episodic_statistics(
            episode=1,
            reward=-1.0,
            batch_index=0,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=1,
            reward=-2.0,
            batch_index=1,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=1,
            reward=-3.0,
            batch_index=2,
        )
        
        self.__reporter.record_episodic_statistics(
            episode=1,
            reward=-4.0,
            batch_index=3,
        )
        
        self.__record_sequential_positive_rewards(episode=2)
        
        stats = self.__get_stats()
        
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].mean_accuracy, 0.0)
        


if __name__ == '__main__':
    unittest.main()