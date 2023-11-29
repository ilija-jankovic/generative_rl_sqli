# Modification of DDPG Keras example from:
# https://keras.io/examples/rl/ddpg_pendulum/

from typing import List
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import tqdm

from .initial_transitions_factory import InitialTransitionsFactory

from .environment import Environment

from .ou_action_noise import OUActionNoise
from .replay_buffer import ReplayBuffer

class DDPG:
    env: Environment
    demonstrations_factory: InitialTransitionsFactory
    lstm_units: int
    batch_size: int
    
    def __init__(self, env: Environment, demonstrations_factory: InitialTransitionsFactory, lstm_units: int, batch_size: int = 2048):
        '''
        `lstm_units` must be at least the length of the environment's dictionary, plus one. The
        additional token placeholder acts as a termination token when generating payloads.
        '''
        assert(lstm_units >= len(env.dictionary) + 1)

        self.env = env
        self.demonstrations_factory = demonstrations_factory
        self.lstm_units = lstm_units
        self.batch_size = batch_size

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def get_actor(self):
        # Add a termination token.
        dictionary_length = len(self.env.dictionary) + 1

        input = layers.Input(shape=(None, 1))
        lstm = layers.LSTM(self.lstm_units, return_state=True)(input)

        # Output of LSTM guide by Jason Brownlee from:
        # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
        state_h = lstm[1]
        state_c = lstm[2]

        state_output = layers.Dense(self.lstm_units, activation='linear')(state_c)
        one_hot_output = layers.Dense(dictionary_length, activation='softmax')(state_h)

        return keras.Model(input, [state_output, one_hot_output])

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.env.state_size,))
        state_out = layers.Dense(16, activation="relu",)(state_input)
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
        
    @tf.function
    def __get_rl_state_lstm_input(self, batch_size, rl_states):
        return tf.reshape(rl_states, [batch_size, self.env.state_size, 1])
    
    @tf.function
    def __get_embeddings(self, indicies: tf.Tensor):
        return tf.gather(self.env.embeddings, tf.cast(indicies, dtype=tf.int32))
    
    @tf.function
    def __get_embedded_lstm_input(self, batch_size, actions, lstm_states):
        embeddings = self.__get_embeddings(actions)
        
        # TODO: Remove this reshaping and process the -1 dimension as LSTM features.
        embeddings = tf.reshape(embeddings, [batch_size, self.env.embedding_size * self.env.action_size])

        input = tf.concat([embeddings, lstm_states], axis=1)
        return tf.expand_dims(input, axis=-1)

    @tf.function
    def __concat_next_token_indicies(self, batch_size, actions, action_index, target: bool, training: bool, rl_states, lstm_states):
        input = tf.cond(
            pred=tf.equal(action_index, 0),
            true_fn=lambda: self.__get_rl_state_lstm_input(batch_size, rl_states),
            false_fn=lambda: self.__get_embedded_lstm_input(batch_size, actions, lstm_states)
        )

        output = tf.cond(
            pred=target,
            true_fn=lambda: self.target_actor(input, training=training),
            false_fn=lambda: self.actor_model(input, training=training))
        
        lstm_states = output[0]
        one_hot = output[1]

        token_indicies = tf.argmax(one_hot, axis=1)
        token_indicies = tf.cast(token_indicies, dtype=tf.float32)

        action_indicies = tf.range(0, batch_size, dtype=tf.float32)

        action_indicies = tf.concat([action_indicies, token_indicies], axis=1)
        
        actions = tf.tensor_scatter_nd_update(actions, action_indicies, token_indicies)
        
        action_index = tf.add(action_index, 1)
        
        return batch_size, actions, action_index, target, training, rl_states, lstm_states

    @tf.function
    def policy(self, states, target: bool, training: bool, batch_size):
        action_size = tf.constant(self.env.action_size, dtype=tf.int32)
        dictionary_length = tf.constant(len(self.env.dictionary), dtype=tf.float32)

        # TODO: Ensure empty token is not part of the dictionary (unnecessary).
        empty_token = dictionary_length - 1.0

        states = tf.convert_to_tensor(states)
        actions = tf.fill([batch_size, action_size], empty_token)
        lstm_states = tf.zeros([batch_size, self.lstm_units])

        action_index = tf.constant(0)

        tf.while_loop(
            cond=lambda *_: True,
            body=self.__concat_next_token_indicies,
            loop_vars=[batch_size, actions, action_index, target, training, states, lstm_states],
            maximum_iterations=action_size,
            shape_invariants=[tf.TensorShape([]), actions.get_shape(), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), states.get_shape(), tf.TensorShape([None, self.lstm_units])]
        )

        return actions
    

    def run(self, total_demonstration_steps: int):
        std_dev = 1.0
        ou_noise = OUActionNoise(mean=np.zeros(self.env.action_size), std_deviation=std_dev * np.ones(self.env.action_size), dt=0.1, theta=0.01)

        actor_model = self.get_actor()
        critic_model = self.get_critic()

        target_actor = self.get_actor()
        target_critic = self.get_critic()

        self.actor_model = actor_model
        self.target_actor = target_actor

        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

        # Learning rate for actor-critic models
        critic_lr = 0.0005
        actor_lr = 0.00025

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        total_episodes = 10000
        # Discount factor for future rewards
        gamma = 0.99
        # Used to update target networks
        tau = 0.005

        buffer = ReplayBuffer(state_size=self.env.state_size, action_size=self.env.action_size,
                              buffer_capacity=50000, batch_size=self.batch_size,
                              actor_model=actor_model,
                              policy=lambda state: self.policy(state, target=False, training=True, batch_size=self.batch_size),
                              target_policy=lambda state: self.policy(state, target=True, training=True, batch_size=self.batch_size),
                              target_critic=target_critic, critic_model=critic_model, actor_optimizer=actor_optimizer,
                              critic_optimizer=critic_optimizer, gamma=gamma)

        '''
        print('Gathering demonstration transitions...')
        
        for obs in tqdm.tqdm(self.demonstrations_factory.gather_transitions(total_demonstration_steps)):
            buffer.record(obs)

        print('Transitions gathered.')
        '''
        print('Running DDPG...')

        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []

        for ep in range(total_episodes):

            prev_state = self.env.create_empty_state()
            episodic_reward = 0
            frame = 0

            while True:
                # Uncomment this to see the Actor in action
                # But not in a python notebook.
                # env.render()

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self.policy([tf_prev_state], target=False, training=False, batch_size=1)[0]
                action += ou_noise()

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

                frame += 1

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode: {}, Avg Reward: {}, Episode Reward: {} Frame Count: {}".format(ep, avg_reward, episodic_reward, frame))
            avg_reward_list.append(avg_reward)
