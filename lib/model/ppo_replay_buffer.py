import tensorflow as tf
import numpy as np

from ..hyperparameters import T, PPO_DEMONSRATION_SAMPLING_PROBABILITY

class PPOReplayBuffer:
    successful_buffer_size: int
    
    __demonstrations_count: int

    __successful_states: np.ndarray
    __successful_actions: np.ndarray
    __successful_rewards: np.ndarray
    __successful_transitions_counter = 1

    # Split into multiple demonstrations for lower rollout.
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

        self.__successful_states[index] = np.array(states)
        self.__successful_actions[index] = np.array(actions)
        self.__successful_rewards[index] = np.array(rewards)

        self.__successful_transitions_counter += 1
        
    def __calculate_uniform_non_demonstration_probabilities(self):
        non_demonstrations_count = self.__successful_transitions_count - self.__demonstrations_count
        
        sampling_probability = (
            1.0 - PPO_DEMONSRATION_SAMPLING_PROBABILITY
        ) / non_demonstrations_count
        
        return [sampling_probability] * (
            self.__successful_transitions_count - self.__demonstrations_count
        )
        
    def __get_sampling_probabilities(self):
        demonstration_probabilities = [
            PPO_DEMONSRATION_SAMPLING_PROBABILITY / self.__demonstrations_count
        ] * self.__demonstrations_count
        
        # Return two sets of uniform probabilities based on demonstration and
        # non-demonstrations.
        #
        # If no non-demonstrations are stored, return a uniform distribution.
        return demonstration_probabilities + self.__calculate_uniform_non_demonstration_probabilities() \
            if self.__successful_transitions_count > self.__demonstrations_count \
            else [1.0 / self.__demonstrations_count] * self.__demonstrations_count

    def sample_successful_trajectories(self, batch_size: int):
        assert(batch_size > 0)

        probabilities = self.__get_sampling_probabilities()
        
        indices = np.random.choice(self.__successful_transitions_count, size=batch_size, p=probabilities)

        states = self.__successful_states[indices]
        actions = self.__successful_actions[indices]
        rewards = self.__successful_rewards[indices]

        return tf.convert_to_tensor(states), \
            tf.convert_to_tensor(actions), \
            tf.convert_to_tensor(rewards)
