import tensorflow as tf
import numpy as np

import model.ppo as ppo

PRIORITY_EXPONENT = 0.3

DEMONSRATION_PROBABILITY = 0.1

class PPOReplayBuffer:
    successful_buffer_size: int

    __successful_states: np.ndarray
    __successful_actions: np.ndarray
    __successful_rewards: np.ndarray
    __successful_transitions_counter = 1

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
        
        assert successful_buffer_size > 1, 'Successful buffer size accounts for reserved demonstration index.'

        assert(demonstrated_successful_states.shape[0] == ppo.T)
        assert(demonstrated_successful_actions.shape[0] == ppo.T)
        assert(demonstrated_successful_rewards.shape[0] == ppo.T)

        self.successful_buffer_size = successful_buffer_size

        self.__successful_states = np.zeros([successful_buffer_size, ppo.T, state_size], dtype=np.float32)
        self.__successful_actions = np.zeros([successful_buffer_size, ppo.T, action_size], dtype=np.int32)
        self.__successful_rewards = np.zeros([successful_buffer_size, ppo.T], dtype=np.float32)

        # Index 0 of successful injections reserved for demonstration.
        self.__successful_states[0] = demonstrated_successful_states
        self.__successful_actions[0] = demonstrated_successful_actions
        self.__successful_rewards[0] = demonstrated_successful_rewards

    @property
    def __successful_transitions_count(self):
        return min(
            self.__successful_transitions_counter,
            self.successful_buffer_size,
        )

    def record_successful_transitions(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
    ):
        assert(states.shape[0] == ppo.T)
        assert(actions.shape[0] == ppo.T)
        assert(rewards.shape[0] == ppo.T)

        # Avoid affecting index 0 as this index contains the demonstration.
        index = self.__successful_transitions_counter % (self.successful_buffer_size - 1) + 1

        self.__successful_states[index] = states
        self.__successful_actions[index] = actions
        self.__successful_rewards[index] = rewards

        self.__successful_transitions_counter += 1

    def sample_successful_trajectories(self, batch_size):
        assert(batch_size > 0)

        # Uniform probabilities except for demonstration.
        probabilities = (
            [DEMONSRATION_PROBABILITY] + (
                [
                    (1.0-DEMONSRATION_PROBABILITY) / (self.__successful_transitions_count-1),
                ] * (self.__successful_transitions_count-1)
            )) \
            if self.__successful_transitions_count > 1 \
            else [1.0]
        
        indices = np.random.choice(self.__successful_transitions_count, size=batch_size, p=probabilities)

        states = self.__successful_states[indices]
        actions = self.__successful_actions[indices]
        rewards = self.__successful_rewards[indices]

        return tf.convert_to_tensor(states), \
            tf.convert_to_tensor(actions), \
            tf.convert_to_tensor(rewards)
