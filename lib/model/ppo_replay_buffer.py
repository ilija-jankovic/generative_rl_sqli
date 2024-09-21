import math
import tensorflow as tf
import numpy as np

from ..hyperparameters import T

class PPOReplayBuffer:
    successful_buffer_size: int
    
    __demonstrations_count: int

    __successful_states: np.ndarray
    __successful_actions: np.ndarray
    __successful_rewards: np.ndarray

    __successful_transitions_counter: int

    __mean_demonstration_reward: int
    '''
    Used to proportionally calculate how many demonstration
    transitions to sample for a replay batch.
    '''


    def __get_clipped_mean_reward(self, rewards):
        # Ensure negative rewards do not affect payload quality
        # proportions.
        clipped_rewards = np.clip(
            rewards,
            a_min=0.0,
            a_max=None,
        )

        return np.mean(clipped_rewards)

    def __init__(
        self,
        state_size: int,
        action_size: int,
        successful_buffer_size: int,
        demonstrated_successful_states: np.ndarray,
        demonstrated_successful_actions: np.ndarray,
        demonstrated_successful_rewards: np.ndarray,
    ):
        assert(state_size > 0)
        assert(action_size > 0)
        
        assert(demonstrated_successful_states.shape[1] == T)
        assert(demonstrated_successful_actions.shape[1] == T)
        assert(demonstrated_successful_rewards.shape[1] == T)
        
        self.__demonstrations_count = demonstrated_successful_states.shape[0]
        
        assert(self.__demonstrations_count >= 1)
        assert(demonstrated_successful_actions.shape[0] == self.__demonstrations_count)
        assert(demonstrated_successful_rewards.shape[0] == self.__demonstrations_count)
        
        assert successful_buffer_size > self.__demonstrations_count, 'Successful buffer size accounts for reserved demonstration indices.'

        self.successful_buffer_size = successful_buffer_size

        self.__successful_states = np.zeros([successful_buffer_size, T, state_size], dtype=np.float64)
        self.__successful_actions = np.zeros([successful_buffer_size, T, action_size], dtype=np.int32)
        self.__successful_rewards = np.zeros([successful_buffer_size, T], dtype=np.float64)

        # Indices up to __demonstrations_count of successful injections reserved
        # for demonstration.
        self.__successful_states[0:self.__demonstrations_count] = demonstrated_successful_states.copy()
        self.__successful_actions[0:self.__demonstrations_count] = demonstrated_successful_actions.copy()
        self.__successful_rewards[0:self.__demonstrations_count] = demonstrated_successful_rewards.copy()

        self.__successful_transitions_counter = successful_buffer_size

        self.__mean_demonstration_reward = self.__get_clipped_mean_reward(
            rewards=demonstrated_successful_rewards,
        )

    @property
    def __successful_transitions_count(self):
        return min(
            self.__successful_transitions_counter,
            self.successful_buffer_size,
        )
        
    def __get_next_successful_buffer_index(self):
        non_demonstrations_count = self.successful_buffer_size - self.__demonstrations_count
        
        # Apply an offset of demonstration count to index, so that demonstrations
        # are preserved.
        return self.__successful_transitions_counter % non_demonstrations_count + \
            self.__demonstrations_count

    def record_successful_transitions(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
    ):
        assert(states.shape[0] == T)
        assert(actions.shape[0] == T)
        assert(rewards.shape[0] == T)

        index = self.__get_next_successful_buffer_index()

        self.__successful_states[index] = np.array(states, dtype=np.float64)
        self.__successful_actions[index] = np.array(actions, dtype=np.float64)
        self.__successful_rewards[index] = np.array(rewards, dtype=np.float64)

        self.__successful_transitions_counter += 1
        
    def __get_demonstration_indices(
        self,
        batch_size: int,
        exploration_rewards: tf.Tensor,
    ):
        assert(self.__mean_demonstration_reward > 0.0)

        mean_exploration_reward = self.__get_clipped_mean_reward(
            rewards=exploration_rewards,
        )
                
        exploration_proportion = np.clip(
            a=mean_exploration_reward / self.__mean_demonstration_reward,
            a_min=None,
            a_max=1.0,
        )
        
        demonstration_batch_size = math.floor((1.0 - exploration_proportion) * batch_size)
        
        return np.random.choice(
            self.__demonstrations_count,
            size=demonstration_batch_size,
        ) if demonstration_batch_size > 0 else np.array([], dtype=np.int32)

    def sample_successful_trajectories(
        self,
        batch_size: int,
        exploration_rewards: tf.Tensor,
    ):
        '''
        Mean exploration reward is proportionally compared against
        max mean demonstration reward to determine the number of
        demonstration transitions to sample for replay batch.
        '''
        assert(batch_size > 0)
        
        demonstration_indices = self.__get_demonstration_indices(
            batch_size=batch_size,
            exploration_rewards=exploration_rewards,
        )
        
        exploration_batch_size = batch_size - demonstration_indices.shape[0]
 
        exploration_indices = np.random.choice(
            self.__successful_transitions_count,
            size=exploration_batch_size,
        ) if exploration_batch_size > 0 else np.array([], dtype=np.int32)
        
        indices = np.concatenate([demonstration_indices, exploration_indices,],)

        states = self.__successful_states[indices]
        actions = self.__successful_actions[indices]
        rewards = self.__successful_rewards[indices]

        return tf.convert_to_tensor(states), \
            tf.convert_to_tensor(actions), \
            tf.convert_to_tensor(rewards)
