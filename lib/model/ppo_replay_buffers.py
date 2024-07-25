import tensorflow as tf
import numpy as np

from .ppo import T

class PPOReplayBuffers:
    successful_buffer_size: int
    max_unsuccessful_buffer_size: int

    __successful_states: np.ndarray
    __successful_actions: np.ndarray
    __successful_rewards: np.ndarray
    __successful_transitions_count = 0

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

        assert(demonstrated_successful_states.shape[0] == T)
        assert(demonstrated_successful_actions.shape[0] == T)
        assert(demonstrated_successful_rewards.shape[0] == T)

        self.successful_buffer_size = successful_buffer_size
        self.max_unsuccessful_buffer_size = unsuccessful_buffer_size

        self.__successful_states = np.full([successful_buffer_size, T, state_size], None)
        self.__successful_actions = np.full([successful_buffer_size, T, action_size], None)
        self.__successful_rewards = np.full([successful_buffer_size, T], None)

        self.__successful_states[0] = demonstrated_successful_states
        self.__successful_actions[0] = demonstrated_successful_actions
        self.__successful_rewards[0] = demonstrated_successful_rewards

        self.__unsuccessful_states = np.full([unsuccessful_buffer_size, T, state_size], None)
        self.__unsuccessful_actions = np.full([unsuccessful_buffer_size, T, action_size], None)
        self.__unsuccessful_rewards = np.full([unsuccessful_buffer_size, T], None)
        self.__unsuccessful_probabilities = np.zeros([unsuccessful_buffer_size])

    def record_successful_transitions(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
    ):
        assert(states.shape[0] == T)
        assert(actions.shape[0] == T)
        assert(rewards.shape[0] == T)

        self.__successful_states = np.array(states) + self.__successful_states[:-1]
        self.__successful_actions = np.array(actions) + self.__successful_actions[:-1]
        self.__successful_rewards = np.array(rewards) + self.__successful_rewards[:-1]

        self.__successful_transitions_count = min(
            self.__successful_transitions_count + 1,
            self.__successful_transitions_count,
        )

        if self.__unsuccessful_states.shape[0] == 0:
            return

        # Remove an unsuccessful buffer slot.
        self.__unsuccessful_states = self.__unsuccessful_states[:-1]
        self.__unsuccessful_actions = self.__unsuccessful_actions[:-1]
        self.__unsuccessful_rewards = self.__unsuccessful_rewards[:-1]
        self.__unsuccessful_probabilities = self.__unsuccessful_probabilities[:-1]

    def record_unsuccessful_transitions(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        value_model: tf.keras.Model,
    ):
        assert(states.shape[0] == T)
        assert(actions.shape[0] == T)
        assert(rewards.shape[0] == T)

        self.__unsuccessful_states = np.array(states) + self.__unsuccessful_states[:-1]
        self.__unsuccessful_actions = np.array(actions) + self.__unsuccessful_actions[:-1]
        self.__unsuccessful_rewards = np.array(rewards) + self.__unsuccessful_rewards[:-1]

        self.__unsuccessful_transitions_count = min(
            self.__unsuccessful_transitions_count + 1,

            # Use a transition component buffer size as current unsuccessful buffer size,
            # as it can be below self.max_unsuccessful_buffer_size if a successful
            # transition was added.
            self.__unsuccessful_states.shape[0],
        )

        priorities = tf.reduce_max(
            tf.map_fn(
                lambda states: value_model(states, training=False),
                self.__unsuccessful_states[:self.__unsuccessful_transitions_count],
            ),
            axis=1,
        )

        alpha = 0.3
        altered_priorities = tf.pow(priorities, alpha)
        probabilites = tf.divide(altered_priorities, tf.reduce_sum(altered_priorities))

        self.__unsuccessful_probabilities[:self.__unsuccessful_transitions_count] = np.array(probabilites)

    def sample_successful_trajectories(self, batch_size):
        assert(batch_size > 0)

        indices = np.random.choice(self.__successful_transitions_count, size=batch_size)

        states = np.take_along_axis(
            self.__successful_states,
            indices=indices,
            axis=0,
        )

        actions = np.take_along_axis(
            self.__successful_actions,
            indices=indices,
            axis=0,
        )

        rewards = np.take_along_axis(
            self.__successful_rewards,
            indices=indices,
            axis=0,
        )

        return tf.convert_to_tensor(states), \
            tf.convert_to_tensor(actions), \
            tf.convert_to_tensor(rewards)
    
    def sample_unsuccessful_trajectories(self, batch_size):
        assert(batch_size > 0)

        indices = np.random.choice(
            self.__unsuccessful_transitions_count,
            size=batch_size,
            p=self.__unsuccessful_probabilities,
        )

        states = np.take_along_axis(
            self.__unsuccessful_states,
            indices=indices,
            axis=0,
        )

        actions = np.take_along_axis(
            self.__unsuccessful_actions,
            indices=indices,
            axis=0,
        )

        rewards = np.take_along_axis(
            self.__unsuccessful_rewards,
            indices=indices,
            axis=0,
        )

        return tf.convert_to_tensor(states), \
            tf.convert_to_tensor(actions), \
            tf.convert_to_tensor(rewards)
