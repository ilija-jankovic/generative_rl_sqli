# Modification of DDPG Keras example from:
# https://keras.io/examples/rl/ddpg_pendulum/

import tensorflow as tf
import keras
from keras import layers
import numpy as np

from .environment import Environment

from .ou_action_noise import OUActionNoise
from .replay_buffer import ReplayBuffer

class DDPG:
    def __init__(self, env: Environment):
        self.env = env

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def get_actor(self, batch_size: int):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        return keras.Sequential([
            layers.LSTM(256, activation="relu", batch_input_shape=(batch_size, self.env.state_size, 1), return_sequences=True),
            layers.LSTM(256, activation="relu", return_sequences=True),
            layers.LSTM(256, activation="relu"),
            layers.Dense(self.env.action_size, activation="tanh", kernel_initializer=last_init)
        ])


    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.env.state_size,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.env.action_size,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model


    def policy(self, state, noise_object, actor_model: keras.Model):
        sampled_actions = tf.squeeze(actor_model(state))
        noise = noise_object()
        
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise


        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, -1.0, 1.0)

        return legal_action


    def run(self):
        std_dev = 1.0
        ou_noise = OUActionNoise(mean=np.zeros(self.env.action_size), std_deviation=float(std_dev) * np.ones(1), dt=0.001)
        batch_size = 4096

        actor_model = self.get_actor(batch_size=batch_size)
        critic_model = self.get_critic()

        target_actor = self.get_actor(batch_size=batch_size)
        target_critic = self.get_critic()

        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        total_episodes = 3000
        # Discount factor for future rewards
        gamma = 0.99
        # Used to update target networks
        tau = 0.005

        buffer = ReplayBuffer(state_size=self.env.state_size, action_size=self.env.action_size,
                              buffer_capacity=50000, batch_size=batch_size, target_actor=target_actor, target_critic=target_critic,
                              actor_model=actor_model, critic_model=critic_model, actor_optimizer=actor_optimizer,
                              critic_optimizer=critic_optimizer, gamma=gamma)


        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []

        for ep in range(total_episodes):

            prev_state = self.env.create_empty_state()
            episodic_reward = 0

            while True:
                # Uncomment this to see the Actor in action
                # But not in a python notebook.
                # env.render()

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self.policy(tf_prev_state, ou_noise, actor_model=actor_model)
                # Recieve state and reward from environment.
                state, reward, done = self.env.perform_action(action) 

                buffer.record((prev_state, action, reward, state))
                episodic_reward += reward

                buffer.learn()
                self.update_target(target_actor.variables, actor_model.variables, tau)
                self.update_target(target_critic.variables, critic_model.variables, tau)

                # End this episode when `done` is True
                if done:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)
