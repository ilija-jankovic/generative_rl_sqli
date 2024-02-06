# Modification of DDPG Keras example from:
# https://keras.io/examples/rl/ddpg_pendulum/

from typing import Callable, List
import tensorflow as tf
import keras
import numpy as np

class ReplayBuffer:
    alpha_priority = 0.3
    epsilon_priority = 0.001
    epsilon_priority_demonstration = 0.001

    def __init__(
            self,
            state_size: int,
            embedding_size: int,
            action_size: int,
            demonstrations_count: int,
            target_policy: Callable[[np.array], np.array],
            target_critic: keras.Model,
            policy: Callable[[np.array], np.array],
            actor_model: keras.Model,
            critic_model: keras.Model,
            actor_optimizer: tf.keras.optimizers.Optimizer,
            critic_optimizer: tf.keras.optimizers.Optimizer,
            buffer_capacity=100000,
            batch_size=64,
            gamma=0.99
        ):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        self.demonstrations_count = demonstrations_count

        self.target_policy = target_policy
        self.target_critic = target_critic
        self.policy = policy
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_size, embedding_size))
        self.action_buffer = np.zeros((self.buffer_capacity, action_size), dtype=np.int32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_size, embedding_size))

        self.priorities_buffer = np.full([self.buffer_capacity], self.epsilon_priority)
        self.priorities_buffer[:self.demonstrations_count] = self.epsilon_priority_demonstration

        self.gamma = gamma

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, epsilon_constants
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_policy(next_state_batch)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)

            td_error = y - critic_value

            critic_loss = tf.math.reduce_mean(tf.math.square(td_error))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables),
        )

        with tf.GradientTape() as tape:
            actions = self.policy(state_batch)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables, unconnected_gradients='zero')
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables), 
        )

        priorities = tf.squeeze(tf.square(td_error)) + actor_loss + epsilon_constants

        return priorities
    

    def __get_replay_probabilities(self, record_range: int):
        priorities = self.priorities_buffer[:record_range]
        priorities = np.power(priorities, self.alpha_priority)

        return priorities / sum(priorities)
        

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)

        probs = self.__get_replay_probabilities(record_range)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size, p=probs)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.int32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)

        epsilon_constants = [
            self.epsilon_priority + self.epsilon_priority_demonstration
                if i < self.demonstrations_count
                else self.epsilon_priority
                for i in batch_indices
            ]
        
        epsilon_constants = tf.convert_to_tensor(epsilon_constants, dtype=tf.float32)

        priorities = self.update(state_batch, action_batch, reward_batch, next_state_batch, epsilon_constants)

        for i in range(self.batch_size):
            buffer_index = self.buffer_counter - self.batch_size - 1 + i
            self.priorities_buffer[buffer_index] = priorities[i]
