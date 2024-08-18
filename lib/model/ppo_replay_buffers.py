import tensorflow as tf
import numpy as np

import model.ppo as ppo

PRIORITY_EXPONENT = 0.3

class PPOReplayBuffers:
    successful_buffer_size: int
    max_unsuccessful_buffer_size: int

    __successful_states: np.ndarray
    __successful_actions: np.ndarray
    __successful_rewards: np.ndarray
    __successful_transitions_count = 1

    __unsuccessful_states: np.ndarray
    __unsuccessful_actions: np.ndarray
    __unsuccessful_rewards: np.ndarray
    __unsuccessful_probabilities: np.ndarray
    __unsuccessful_transitions_count = 0

    def __init__(
        self,
        state_size: int,
        action_size: int,
        successful_buffer_size: int,
        unsuccessful_buffer_size: int,
        demonstrated_successful_states: np.ndarray,
        demonstrated_successful_actions: np.ndarray,
        demonstrated_successful_rewards: np.ndarray,
    ):
        assert(state_size > 0)
        assert(action_size > 0)
        assert(successful_buffer_size > 0)
        assert(unsuccessful_buffer_size > 0)

        assert(demonstrated_successful_states.shape[0] == ppo.T)
        assert(demonstrated_successful_actions.shape[0] == ppo.T)
        assert(demonstrated_successful_rewards.shape[0] == ppo.T)

        self.successful_buffer_size = successful_buffer_size
        self.max_unsuccessful_buffer_size = unsuccessful_buffer_size

        self.__successful_states = np.zeros([successful_buffer_size, ppo.T, state_size], dtype=np.float32)
        self.__successful_actions = np.zeros([successful_buffer_size, ppo.T, action_size], dtype=np.int32)
        self.__successful_rewards = np.zeros([successful_buffer_size, ppo.T], dtype=np.float32)

        self.__successful_states[0] = demonstrated_successful_states
        self.__successful_actions[0] = demonstrated_successful_actions
        self.__successful_rewards[0] = demonstrated_successful_rewards

        self.__unsuccessful_states = np.zeros([unsuccessful_buffer_size, ppo.T, state_size], dtype=np.float32)
        self.__unsuccessful_actions = np.zeros([unsuccessful_buffer_size, ppo.T, action_size], dtype=np.int32)
        self.__unsuccessful_rewards = np.zeros([unsuccessful_buffer_size, ppo.T], dtype=np.float32)
        self.__unsuccessful_probabilities = np.zeros([unsuccessful_buffer_size], dtype=np.float64)

    def is_unsuccessful_buffer_empty(self):
        return self.__unsuccessful_transitions_count == 0

    def record_successful_transitions(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
    ):
        assert(states.shape[0] == ppo.T)
        assert(actions.shape[0] == ppo.T)
        assert(rewards.shape[0] == ppo.T)

        self.__successful_states = np.insert(self.__successful_states[:-1], 0, states, axis=0)
        self.__successful_actions = np.insert(self.__successful_actions[:-1], 0, actions, axis=0)
        self.__successful_rewards = np.insert(self.__successful_rewards[:-1], 0, rewards, axis=0)

        self.__successful_transitions_count = min(
            self.__successful_transitions_count + 1,
            self.successful_buffer_size,
        )

        if self.__unsuccessful_states.shape[0] == 0:
            return

        # Remove an unsuccessful buffer slot.
        self.__unsuccessful_states = self.__unsuccessful_states[:-1]
        self.__unsuccessful_actions = self.__unsuccessful_actions[:-1]
        self.__unsuccessful_rewards = self.__unsuccessful_rewards[:-1]
        self.__unsuccessful_probabilities = self.__unsuccessful_probabilities[:-1]

        if self.__unsuccessful_transitions_count > self.__unsuccessful_states.shape[0]:
            self.__unsuccessful_transitions_count -= 1

    def record_unsuccessful_transitions(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        value_model: tf.keras.Model,
    ):
        assert(states.shape[0] == ppo.T)
        assert(actions.shape[0] == ppo.T)
        assert(rewards.shape[0] == ppo.T)

        self.__unsuccessful_states = np.insert(self.__unsuccessful_states[:-1], 0, states, axis=0)
        self.__unsuccessful_actions = np.insert(self.__unsuccessful_actions[:-1], 0, actions, axis=0)
        self.__unsuccessful_rewards = np.insert(self.__unsuccessful_rewards[:-1], 0, rewards, axis=0)

        self.__unsuccessful_transitions_count = min(
            self.__unsuccessful_transitions_count + 1,

            # Use a transition component buffer size as current unsuccessful buffer size,
            # as it can be below self.max_unsuccessful_buffer_size if a successful
            # transition was added.
            self.__unsuccessful_states.shape[0],
        )

        priorities = tf.reduce_max([
            value_model(tf.convert_to_tensor(states), training=False)
                for states in self.__unsuccessful_states[:self.__unsuccessful_transitions_count]
            ],
            axis=1,
        )

        priorities = tf.squeeze(priorities)

        # Casting to greater floating point precision matches probabilities type, which can contain
        # very low probabilities.
        priorities = tf.cast(priorities, dtype=tf.float64)

        altered_priorities = tf.pow(priorities, PRIORITY_EXPONENT)
        probabilites = tf.math.divide_no_nan(altered_priorities, tf.reduce_sum(altered_priorities))

        # Ensure probabilities add to NumPy's tolerance of deviance from one.
        #
        # Inspired by Divakar's solution from:
        # https://stackoverflow.com/a/43644348
        probabilites = tf.divide(probabilites, tf.reduce_sum(probabilites))

        self.__unsuccessful_probabilities[:self.__unsuccessful_transitions_count] = np.array(probabilites, dtype=np.float64)

    def sample_successful_trajectories(self, batch_size):
        assert(batch_size > 0)

        indices = np.random.choice(self.__successful_transitions_count, size=batch_size)

        states = self.__successful_states[indices]
        actions = self.__successful_actions[indices]
        rewards = self.__successful_rewards[indices]

        return tf.convert_to_tensor(states), \
            tf.convert_to_tensor(actions), \
            tf.convert_to_tensor(rewards)
    
    def sample_unsuccessful_trajectories(self, batch_size):
        assert(batch_size > 0)
        assert(self.__unsuccessful_transitions_count > 0)

        probabilities = self.__unsuccessful_probabilities[:self.__unsuccessful_transitions_count]

        indices = np.random.choice(
            self.__unsuccessful_transitions_count,
            size=batch_size,
            p=probabilities,
        )

        states = self.__unsuccessful_states[indices]
        actions = self.__unsuccessful_actions[indices]
        rewards = self.__unsuccessful_rewards[indices]

        return tf.convert_to_tensor(states), \
            tf.convert_to_tensor(actions), \
            tf.convert_to_tensor(rewards)
