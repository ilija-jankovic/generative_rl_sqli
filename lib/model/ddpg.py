# Modification of DDPG Keras example from:
# https://keras.io/examples/rl/ddpg_pendulum/

import datetime
import math
import random
from typing import List
import tensorflow as tf
import keras
from keras import layers
import numpy as np
from sqltree import sqltree

from .ddpg_hyperparameters import DDPGHyperparameters
from .ddpg_running_statistic import DDPGRunningStatistic
from .ddpg_payload_statistic import DDPGPayloadStatistic
from .reporter import Reporter

from .enums.policy_type import PolicyType
from .environment import Environment
from .replay_buffer import ReplayBuffer

class DDPG:
    env: Environment
    encoded_payloads: List[List[int]]
    params: DDPGHyperparameters
    actor_lstm_units: int
    
    actor_perturbed: keras.Model

    __epsilon: float
    
    def __init__(self, env: Environment, encoded_payloads: List[List[int]], params: DDPGHyperparameters, actor_lstm_units: int = 1024):
        assert(params.psi >= 0.0 and params.psi <= 1.0)

        # Ensure last token in dictionary is the empty token.
        assert(len(env.dictionary) > 0 and env.dictionary[-1] == '')

        self.env = env
        self.params = params
        self.actor_lstm_units = actor_lstm_units

        self.__epsilon = params.epsilon_start

        # Take out empty tokens.
        encoded_payloads = [[token for token in payload if token != len(env.dictionary) - 1] for payload in encoded_payloads]

        # Filter all payloads within action size.
        encoded_payloads = list(filter(lambda payload: len(payload) <= self.env.action_size, encoded_payloads))

        # Pad with min int.
        self.encoded_payloads = [[payload[i] if i < len(payload) else tf.int32.min for i in range(self.env.action_size)] for payload in encoded_payloads]

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    @tf.function
    def get_mask(self, payload):
        dictionary_length = len(self.env.dictionary)
        suffix_length = len(self.params.suffix)

        mask = []
        
        for i in range(dictionary_length - 1):
            token = self.env.dictionary[i]

            try:
                sqltree(f'SELECT * FROM products WHERE id= \'{payload[:-suffix_length] + token + self.params.suffix}\'')
                mask.append(1.0)
            except:
                mask.append(1.0 - self.params.psi)

        # Account for termination token.
        mask += [1.0]

        return tf.stack(mask)

    @tf.function
    def mask_one_hot_encoding(self, single_one_hot_encoding, action: tf.Tensor):
        payload = tf.py_function(self.env.get_payload, [action], tf.string)

        # Avoid unnecessary mask construction if psi is zero, as the mask will always
        # be ones in this case.
        return tf.cond(tf.less_equal(self.params.psi, 0.0),
            true_fn=lambda: single_one_hot_encoding,
            false_fn=lambda: single_one_hot_encoding * self.get_mask(payload))
    
    @tf.function
    def get_embeddings(self, one_hot_encoding, actions):        
        one_hot_encoding = [self.mask_one_hot_encoding(one_hot_encoding[i], actions[i]) for i in range(self.params.batch_size)]
        one_hot_encoding = tf.convert_to_tensor(one_hot_encoding, dtype=tf.float32)

        indices = tf.argmax(one_hot_encoding, axis=1, output_type=tf.int32)
        embeddings = tf.convert_to_tensor(self.env.embeddings, dtype=tf.float32)
        
        return indices, tf.gather(embeddings, indices)

    def get_actor(self):
        DROPOUT = 0.1
        C_PADDING = self.env.embedding_size - (self.actor_lstm_units % self.env.embedding_size)

        dictionary_length = len(self.env.dictionary)

        input_lstm = layers.Input(shape=(None, self.env.embedding_size), batch_size=self.params.batch_size)

        lstm = layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True, kernel_regularizer='L2')(input_lstm)
        lstm = layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True, kernel_regularizer='L2')(lstm)
        lstm = layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, kernel_regularizer='L2')(lstm)

        # Output of LSTM guide by Jason Brownlee from:
        # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
        state_h = lstm[1]
        state_c = lstm[2]

        # Add normalisation layers between perturbed layers (pg. 3).
        dense = layers.Dropout(DROPOUT)(state_h)
        dense = layers.Dense(1024, activation='relu')(dense)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(DROPOUT)(dense)
        dense = layers.Dense(1024, activation='relu')(dense)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(DROPOUT)(dense)
        dense_output = layers.Dense(dictionary_length, activation='softmax')(dense)

        padded_state_c = layers.Lambda(lambda state_c: tf.pad(state_c, [[0, 0], [0, C_PADDING]]))(state_c)

        input_actions = layers.Input(shape=(self.env.action_size), batch_size=self.params.batch_size, dtype=tf.int32)
        indices_output, embedding_output = layers.Lambda(lambda input: self.get_embeddings(input[0], input[1]))((dense_output, input_actions))

        return keras.Model([input_lstm, input_actions], [padded_state_c, indices_output, embedding_output])

    def get_critic(self):
        LSTM_UNITS = 512

        # State as input
        state_input = layers.Input(shape=(self.env.state_size, self.env.embedding_size), batch_size=self.params.batch_size)

        lstm = layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True)(state_input)
        lstm = layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True)(lstm)
        lstm_state_out = layers.CuDNNLSTM(LSTM_UNITS)(lstm)

        # Action as input
        action_input = layers.Input(shape=(self.env.action_size, 1), batch_size=self.params.batch_size)

        lstm = layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True)(action_input)
        lstm = layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True)(lstm)
        lstm_action_out = layers.CuDNNLSTM(LSTM_UNITS)(lstm)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([lstm_state_out, lstm_action_out])

        out = layers.Dense(1024, activation="relu")(concat)
        out = layers.Dense(1024, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    @tf.function
    def get_embedded_lstm_input(self, embeddings, lstm_states):
        input = tf.concat([embeddings, lstm_states], axis=1)

        return tf.reshape(input, [self.params.batch_size, -1, self.env.embedding_size])

    @tf.function
    def concat_next_token_indicies(self, actions, action_index, action_index_float, embeddings, type: int, training: bool, rl_states, lstm_states):
        batch_size = self.params.batch_size

        input = tf.cond(
            pred=tf.equal(action_index, 0),
            true_fn=lambda: (rl_states, actions),
            false_fn=lambda: (self.get_embedded_lstm_input(embeddings, lstm_states), actions)
        )

        output = tf.cond(
            tf.equal(type, PolicyType.NORMAL.value),
                true_fn=lambda: self.actor_model(input, training=training),
                false_fn=lambda: tf.cond(
                tf.equal(type, PolicyType.PERTURBED.value),
                    true_fn=lambda: self.actor_perturbed(input, training=training),
                    false_fn=lambda: self.target_actor(input, training=training)))

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
        
        return actions, action_index, action_index_float, embeddings, type, training, rl_states, lstm_states

    @tf.function
    def policy(self, states, type: int, training: bool):
        '''
        `type` is expected to be the enumerated value of a `PolicyType`.

        This enum cannot be passed directly due to `@tf.function` limitations.
        '''
        batch_size = self.params.batch_size

        action_size = tf.constant(self.env.action_size, dtype=tf.int32)

        actions = tf.fill([batch_size, action_size], len(self.env.dictionary) - 1)
        
        embeddings = tf.zeros([batch_size, self.env.embedding_size], dtype=tf.float32)
        lstm_states = tf.zeros([batch_size, self.actor_lstm_units + self.env.embedding_size - (self.actor_lstm_units % self.env.embedding_size)], dtype=tf.float32)

        action_index = tf.constant(0, dtype=tf.int32)
        action_index_float = tf.constant(0.0, dtype=tf.float32)

        actions, *_ = tf.while_loop(
            cond=lambda *_: True,
            body=self.concat_next_token_indicies,
            loop_vars=(actions, action_index, action_index_float, embeddings, type, training, states, lstm_states),
            maximum_iterations=action_size,
        )

        return actions


    def __run_action(self, action: tf.Tensor, prev_state: tf.Tensor, buffer: ReplayBuffer):
        # Recieve state and reward from environment.
        state, reward, done = self.env.perform_action(action)

        buffer.record((prev_state, action, reward, state))

        return state, reward, done
    
    
    def __update_perturbed_actor(self):
        self.actor_perturbed.set_weights(self.actor_model.get_weights()) 

        # Adding noise to model weights algorithm by Daan Klijn:
        # https://medium.com/adding-noise-to-network-weights-in-tensorflow/adding-noise-to-network-weights-in-tensorflow-fddc82e851cb
        for layer in self.actor_perturbed.trainable_weights:
            noise = np.random.normal(loc=0.0, scale=self.__epsilon, size=layer.shape)
            layer.assign_add(noise)
            
    
    def __get_perturbed_actions(self, states: tf.Tensor):
        actions = self.policy(states, PolicyType.PERTURBED.value, training=False)

        self.__epsilon = max(self.__epsilon * self.params.epsilon_decay, self.params.epsilon_min)

        return actions
    

    def run(self, run_demonstrations: bool):
        actor_model = self.get_actor()
        actor_perturbed = self.get_actor()
        critic_model = self.get_critic()

        target_actor = self.get_actor()
        target_critic = self.get_critic()

        self.actor_model = actor_model
        self.actor_perturbed = actor_perturbed
        self.target_actor = target_actor

        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

        # Nadam for RNNs recommended by OverLordGoldDragon:
        # https://stackoverflow.com/questions/48714407/rnn-regularization-which-component-to-regularize/58868383#58868383
        critic_optimizer = tf.keras.optimizers.Nadam(self.params.critic_learning_rate)
        actor_optimizer = tf.keras.optimizers.Nadam(self.params.actor_learning_rate)

        total_episodes = 500

        buffer = ReplayBuffer(
            state_size=self.env.state_size,
            embedding_size=self.env.embedding_size,
            action_size=self.env.action_size,
            buffer_capacity=self.params.buffer_size,
            batch_size=self.params.batch_size,
            actor_model=actor_model,
            policy=lambda state: self.policy(state, PolicyType.NORMAL.value, training=True),
            target_policy=lambda state: self.policy(state, PolicyType.TARGET.value, training=True),
            target_critic=target_critic,
            critic_model=critic_model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=self.params.gamma
        )

        avg_batch_rewards = []
        
        demonstrations_completed = 0
        frame = 0

        total_demonstration_steps = math.ceil(len(self.encoded_payloads) / self.params.batch_size) * self.params.batch_size * 2

        run_demonstrations = run_demonstrations and self.encoded_payloads is not None

        reporter = Reporter()

        print('Starting reporter...')
        reporter.start(self.params)

        if run_demonstrations:
            print('Gathering demonstrations...')

        for ep in range(1, total_episodes + 1):
            prev_states = [self.env.create_empty_state(index=float(i)) for i in range(self.params.batch_size)]
            prev_states = tf.convert_to_tensor(prev_states)

            # Update perturbed actor at beginning of episode for stability.
            # (Pg. 3 of Adapative Parameter Space Noise paper).
            self.__update_perturbed_actor()

            while True:
                demonstrate = run_demonstrations and demonstrations_completed < total_demonstration_steps

                prev_epsilon = self.__epsilon

                if demonstrate:
                    actions = tf.convert_to_tensor([random.choice(self.encoded_payloads) for _ in range(self.params.batch_size)]), 0.0
                    demonstrations_completed += self.params.batch_size
                    
                    print(f'{demonstrations_completed}/{total_demonstration_steps} demonstration observations gathered.')

                    if demonstrations_completed >= total_demonstration_steps:
                        print('Transitions gathered.')
                else:
                   actions = self.__get_perturbed_actions(prev_states)

                env_tuples = [self.__run_action(actions[i], prev_states[i], buffer) for i in range(len(actions))]

                states = tf.convert_to_tensor([env_tuple[0] for env_tuple in env_tuples])

                avg_batch_reward = sum([env_tuple[1] for env_tuple in env_tuples]) / self.params.batch_size
                avg_batch_rewards.append(avg_batch_reward)
                
                done = True in [env_tuple[2] for env_tuple in env_tuples]

                frame += self.params.batch_size

                avg_reward = np.mean(avg_batch_rewards)

                running_stat = DDPGRunningStatistic(
                    epsiode=ep,
                    frame=frame,
                    total_avg_reward=avg_reward,
                    is_demonstration=demonstrate,
                    epsilon=prev_epsilon
                )
                
                reporter.record_running_statistic(running_stat)

                payload_stats = [
                    DDPGPayloadStatistic(
                        epsiode=ep,
                        frame=frame,
                        payload=self.env.get_payload(actions[i]),
                        reward=env_tuples[i][1],
                        is_demonstration=demonstrate
                    ) for i in range(len(actions)) if env_tuples[i][1] > 0.0
                ]
                
                for stat in payload_stats:
                    reporter.record_payload_statistic(stat)

                buffer.learn()
                self.update_target(target_actor.variables, actor_model.variables, self.params.tau)
                self.update_target(target_critic.variables, critic_model.variables, self.params.tau)

                # End this episode when `done` is True
                if done:
                    break

                prev_states = states

            print("[{}] Episode: {}, Avg Reward: {}, Total Frame Count: {}".format(datetime.datetime.now(), ep, avg_reward, frame))
