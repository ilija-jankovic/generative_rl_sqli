# Modification of DDPG Keras example from:
# https://keras.io/examples/rl/ddpg_pendulum/

from typing import List
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import tqdm
from sqltree import sqltree

from .initial_transitions_factory import InitialTransitionsFactory

from .environment import Environment

from .ou_action_noise import OUActionNoise
from .replay_buffer import ReplayBuffer

class DDPG:
    env: Environment
    demonstrations_factory: InitialTransitionsFactory
    lstm_units: int
    psi: float
    
    def __init__(self, env: Environment, demonstrations_factory: InitialTransitionsFactory, lstm_units: int, psi: float = 0.0):
        assert(psi >= 0.0 and psi <= 1.0)

        self.env = env
        self.demonstrations_factory = demonstrations_factory
        self.lstm_units = lstm_units
        self.psi = psi

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    @tf.function
    def __get_mask(self, payload):
        dictionary_length = len(self.env.dictionary)

        mask = []
        
        for i in range(dictionary_length):
            token = self.env.dictionary[i]

            try:
                sqltree(payload + token)
                mask.append(1.0)
            except:
                mask.append(1.0 - self.psi)

        # Account for termination tokens.
        mask += (self.lstm_units - dictionary_length) * [1.0]

        return tf.stack(mask)

    @tf.function
    def __mask_one_hot_encoding(self, single_one_hot_encoding, action: tf.Tensor):
        payload = tf.py_function(self.env.get_payload, [action], tf.string)

        return single_one_hot_encoding * self.__get_mask(payload)
    
    @tf.function
    def __get_embeddings(self, one_hot_encoding, actions):
        one_hot_encoding = [self.__mask_one_hot_encoding(one_hot_encoding[i], actions[i]) for i in range(self.env.batch_size)]
        one_hot_encoding = tf.convert_to_tensor(one_hot_encoding, dtype=tf.float32)

        indices = tf.argmax(one_hot_encoding, axis=1, output_type=tf.int32)
        embeddings = tf.concat([tf.convert_to_tensor(self.env.embeddings, dtype=tf.float32), tf.zeros((self.lstm_units - len(self.env.dictionary), self.env.embedding_size))], axis=0)
        
        return indices, tf.gather(embeddings, indices)

    def get_actor(self):
        C_PADDING = self.env.embedding_size - (self.lstm_units % self.env.embedding_size)

        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        input_lstm = layers.Input(shape=(None, self.env.embedding_size), batch_size=self.env.batch_size)
        lstm = layers.LSTM(self.lstm_units, return_state=True, activation='softmax', kernel_initializer=last_init)(input_lstm)

        # Output of LSTM guide by Jason Brownlee from:
        # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
        state_h = lstm[1]
        state_c = lstm[2]
        
        #one_hot_encoding = layers.Dense(self.lstm_units, activation='sigmoid')(state_h)

        padded_state_c = layers.Lambda(lambda state_c: tf.pad(state_c, [[0, 0], [0, C_PADDING]]))(state_c)

        input_actions = layers.Input(shape=(self.env.action_size), batch_size=self.env.batch_size, dtype=tf.int32)
        indices_output, embedding_output = layers.Lambda(lambda input: self.__get_embeddings(input[0], input[1]))((state_h, input_actions))

        return keras.Model([input_lstm, input_actions], [padded_state_c, indices_output, embedding_output])

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.env.state_size, self.env.embedding_size))
        state_flatten = layers.Flatten()(state_input)
        state_out = layers.Dense(16, activation="relu",)(state_flatten)
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

    
    # Concat appears to break gradients:
    # https://github.com/tensorflow/tensorflow/issues/37726#issuecomment-1101018112
    @tf.function
    def __get_embedded_lstm_input(self, embeddings, lstm_states):
        input = tf.concat([embeddings, lstm_states], axis=1)

        return tf.reshape(input, [self.env.batch_size, -1, self.env.embedding_size])

    @tf.function
    def __concat_next_token_indicies(self, actions, action_index, action_index_float, embeddings, target: bool, training: bool, rl_states, lstm_states):
        batch_size = self.env.batch_size

        input = tf.cond(
            pred=tf.equal(action_index, 0),
            true_fn=lambda: (rl_states, actions),
            false_fn=lambda: (self.__get_embedded_lstm_input(embeddings, lstm_states), actions)
        )

        output = tf.cond(
            pred=target,
            true_fn=lambda: self.target_actor(input, training=training),
            false_fn=lambda: self.actor_model(input, training=training))

        lstm_states = output[0]
        indices = output[1]

        # action_index_float is the length of the action after incrementing, which
        # is then used in the below embedding average calculation.
        action_index_float = tf.add(action_index_float, 1.0)

        # Adding to an average solution by Damien and Dan Dascalescu from:
        # https://math.stackexchange.com/questions/22348/how-to-add-and-subtract-values-from-an-average
        embeddings = embeddings + (output[2] - embeddings) / action_index_float

        action_indices = tf.range(0, batch_size, dtype=tf.int32)
        action_indices = tf.expand_dims(action_indices, axis=1)
        action_indices = tf.pad(action_indices, [[0, 0], [0, 1]], constant_values=action_index)

        actions = tf.tensor_scatter_nd_update(actions, action_indices, indices)

        action_index = tf.add(action_index, 1)
        
        return actions, action_index, action_index_float, embeddings, target, training, rl_states, lstm_states

    @tf.function
    def policy(self, states, target: bool, training: bool):
        batch_size = self.env.batch_size

        action_size = tf.constant(self.env.action_size, dtype=tf.int32)

        actions = tf.fill([batch_size, action_size], tf.int32.min)
        
        embeddings = tf.zeros([batch_size, self.env.embedding_size], dtype=tf.float32)
        lstm_states = tf.zeros([batch_size, self.lstm_units + self.env.embedding_size - (self.lstm_units % self.env.embedding_size)], dtype=tf.float32)

        action_index = tf.constant(0, dtype=tf.int32)
        action_index_float = tf.constant(0.0, dtype=tf.float32)

        actions, *_ = tf.while_loop(
            cond=lambda *_: True,
            body=self.__concat_next_token_indicies,
            loop_vars=(actions, action_index, action_index_float, embeddings, target, training, states, lstm_states),
            maximum_iterations=action_size,
        )

        return actions
    

    def __run_action(self, action: tf.Tensor, prev_state: tf.Tensor, buffer: ReplayBuffer, ou_noise: OUActionNoise):
        print(action)
        action += ou_noise()

        # Recieve state and reward from environment.
        state, reward, done = self.env.perform_action(action)

        buffer.record((prev_state, action, reward, state))

        return state, reward, done
    

    def run(self, total_demonstration_steps: int):
        batch_size = self.env.batch_size

        std_dev = len(self.env.dictionary) * 0.2
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
        actor_lr = 0.0025

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        total_episodes = 500
        # Discount factor for future rewards
        gamma = 0.98
        # Used to update target networks
        tau = 0.005

        buffer = ReplayBuffer(state_size=self.env.state_size, embedding_size=self.env.embedding_size, action_size=self.env.action_size,
                              buffer_capacity=1000000, batch_size=batch_size,
                              actor_model=actor_model,
                              policy=lambda state: self.policy(state, target=False, training=True),
                              target_policy=lambda state: self.policy(state, target=True, training=True),
                              target_critic=target_critic, critic_model=critic_model, actor_optimizer=actor_optimizer,
                              critic_optimizer=critic_optimizer, gamma=gamma)

        if self.demonstrations_factory is None:
            print('Skipping demonstrations...')
        else:
            print('Gathering demonstration transitions...')
            
            for obs in tqdm.tqdm(self.demonstrations_factory.gather_transitions(total_demonstration_steps)):
                buffer.record(obs)

            print('Transitions gathered.')
            print('Running DDPG...')

        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []

        for ep in range(total_episodes):

            prev_states = [self.env.create_empty_state() for _ in range(self.env.batch_size)]
            prev_states = tf.convert_to_tensor(prev_states)

            episodic_reward = 0
            frame = 0

            while True:
                #print(target_actor.trainable_variables[0])
                actions = self.policy(prev_states, target=False, training=False)

                env_tuples = [self.__run_action(actions[i], prev_states[i], buffer, ou_noise) for i in range(len(actions))]

                states = tf.convert_to_tensor([env_tuple[0] for env_tuple in env_tuples])

                episodic_reward += sum([env_tuple[1] for env_tuple in env_tuples])
                done = True in [env_tuple[2] for env_tuple in env_tuples]

                frame += self.env.batch_size

                buffer.learn()
                self.update_target(target_actor.variables, actor_model.variables, tau)
                self.update_target(target_critic.variables, critic_model.variables, tau)

                # End this episode when `done` is True
                if done:
                    break

                prev_states = states

            ep_reward_list.append(episodic_reward)

            avg_reward = np.mean(ep_reward_list)
            print("Episode: {}, Avg Reward: {}, Episode Reward: {} Frame Count: {}".format(ep + 1, avg_reward, episodic_reward, frame))
            avg_reward_list.append(avg_reward)
