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

    __stddev: float
    __epsilon: float
    
    def __init__(self, env: Environment, encoded_payloads: List[List[int]], params: DDPGHyperparameters, actor_lstm_units: int = 512):
        assert(params.psi >= 0.0 and params.psi <= 1.0)

        dictionary_length = len(env.dictionary)

        # Ensure last token in dictionary is the empty token.
        assert(dictionary_length > 0 and env.dictionary[-1] == '')

        self.env = env
        self.params = params
        self.actor_lstm_units = actor_lstm_units


        self.__stddev = params.starting_stddev
        
        self.__epsilon = None if self.params.constant_stddev else params.epsilon_start 

        # Take out empty tokens.
        encoded_payloads = [[token for token in payload if token != len(env.dictionary) - 1] for payload in encoded_payloads]

        # Filter all payloads within action size.
        encoded_payloads = list(filter(lambda payload: len(payload) <= self.env.action_size, encoded_payloads))

        # Pad with min int.
        self.encoded_payloads = [[payload[i] if i < len(payload) else dictionary_length - 1 for i in range(self.env.action_size)] for payload in encoded_payloads]

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
        C_PADDING = self.env.embedding_size - (self.actor_lstm_units % self.env.embedding_size)

        dictionary_length = len(self.env.dictionary)

        input_lstm = layers.Input(shape=(None, self.env.embedding_size), batch_size=self.params.batch_size)

        lstm = layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True)(input_lstm)
        lstm = layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True)(lstm)
        lstm = layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True)(lstm)
        lstm = layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True)(lstm)
        lstm = layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True)(lstm)

        # Output of LSTM guide by Jason Brownlee from:
        # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
        state_h = lstm[1]
        state_c = lstm[2]

        # Add normalisation layers between perturbed layers (pg. 3).
        dense = layers.Dense(1024, activation='relu')(state_h)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dense(1024, activation='relu')(dense)
        dense = layers.BatchNormalization()(dense)
        dense_output = layers.Dense(dictionary_length, activation='softmax')(dense)

        padded_state_c = layers.Lambda(lambda state_c: tf.pad(state_c, [[0, 0], [0, C_PADDING]]))(state_c)

        input_actions = layers.Input(shape=(self.env.action_size), batch_size=self.params.batch_size, dtype=tf.int32)
        indices_output, embedding_output = layers.Lambda(lambda input: self.get_embeddings(input[0], input[1]))((dense_output, input_actions))

        return keras.Model([input_lstm, input_actions], [padded_state_c, indices_output, embedding_output])

    def get_critic(self):
        LSTM_UNITS = 1024

        action_input = layers.Input(shape=(self.env.action_size, 1), batch_size=self.params.batch_size)

        # TODO: Bidirectional suited for NLP sequences.
        lstm = layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True)(action_input)
        lstm = layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True)(lstm)
        lstm = layers.CuDNNLSTM(LSTM_UNITS)(lstm)

        out = layers.Dense(1024, activation="relu")(lstm)
        out = layers.Dense(1024, activation="relu")(out)
        out = layers.Dense(512, activation="relu")(out)
        out = layers.Dense(256, activation="relu")(out)
        out = layers.Dense(128, activation="relu")(out)
        out = layers.Dense(64, activation="relu")(out)
        out = layers.Dense(32, activation="relu")(out)
        out = layers.Dense(16, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([action_input], outputs)

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


    def __run_actions(self, actions: tf.Tensor, prev_states: tf.Tensor, buffer: ReplayBuffer, ignore_episode: bool, is_demonstration: bool):
        # Recieve state and reward from environment.
        env_tuples = [self.env.perform_action(action, ignore_episode=ignore_episode) for action in actions]

        obs_tuples = [(prev_states[i], actions[i], env_tuples[i][1], env_tuples[i][0]) for i in range(self.params.batch_size)]

        buffer.record(obs_tuples, is_demonstration=is_demonstration)

        return env_tuples
    
    
    def __update_perturbed_actor(self):
        self.actor_perturbed.set_weights(self.actor_model.get_weights()) 

        # Adding noise to model weights algorithm by Daan Klijn:
        # https://medium.com/adding-noise-to-network-weights-in-tensorflow/adding-noise-to-network-weights-in-tensorflow-fddc82e851cb
        for layer in self.actor_perturbed.trainable_weights:
            noise = np.random.normal(loc=0.0, scale=self.__stddev, size=layer.shape)
            layer.assign_add(noise)


    @tf.function
    def normalize_0_1(self, tensor: tf.Tensor):
        '''
        Expects only non-negative values in `tensor`.
        '''
        tf.debugging.assert_non_negative(tensor, '[0,1] normalization ' +
            'tensor must not contain negative values.')
        
        sums = tf.reduce_sum(tensor, axis=-1, keepdims=True)
        
        return tensor / sums


    @tf.function
    def get_kl_divergence(self, t1: tf.Tensor, t2: tf.Tensor):
        '''
        `t1` and `t2` are normalized between `[0,1]` for divergence calculation,
        but must only contain non-negative values.
        '''
        tf.debugging.assert_non_negative(t1,'Tensor for divergence ' +
            'calculation must not contain negative values.')
        tf.debugging.assert_non_negative(t2,'Tensor for divergence ' +
            'calculation must not contain negative values.')

        t1 = tf.cast(t1, dtype=tf.float32)
        t2 = tf.cast(t2, dtype=tf.float32)

        # Ensure values are scaled to sum to one to meet KL divergence
        # requirement of probability distribution inputs.
        t1 = self.normalize_0_1(t1)
        t2 = self.normalize_0_1(t2)

        divergences = tf.keras.metrics.kl_divergence(t1, t2)

        return tf.reduce_mean(divergences)


    # Page 15 of Adaptive Noise Paper for epsilon-based KL divergence threshold.
    def __calculate_distance_threshold(self):
        return -np.log(1.0 - self.__epsilon + (self.__epsilon / self.params.batch_size))


    def __iterate_adaptive_stddev(self, divergence: float, distance_threshold: float):
        # Adapted solution by Sören Kirchner:
        # https://soeren-kirchner.medium.com/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3
        #
        # Changed distance and sigma definitions based on the
        # PARAMETER SPACE NOISE FOR EXPLORATION paper:
        # https://openreview.net/pdf?id=ByBAl2eAZ
        #
        # Page 15:
        # "Setting δ := σ as
        # the adaptive parameter space threshold thus results in effective action space noise that has the same
        # standard deviation as regular Gaussian action space noise."
        if divergence <= distance_threshold:
            self.__stddev *= self.params.alpha_scalar
        else:
            self.__stddev /= self.params.alpha_scalar
            
    
    def __get_perturbed_actions(self, states: tf.Tensor):
        actions = self.policy(states, PolicyType.NORMAL.value, training=False)
        actions_perturbed = self.policy(states, PolicyType.PERTURBED.value, training=False)

        constant_stddev = self.params.constant_stddev

        # Actions are comprised of indices which are never negative, meeting
        # the conditions of the KL divergence method.
        divergence = self.get_kl_divergence(actions, actions_perturbed)

        distance_threshold = None if constant_stddev else self.__calculate_distance_threshold()

        if not constant_stddev:
            self.__iterate_adaptive_stddev(divergence=divergence, distance_threshold=distance_threshold)

        return actions_perturbed, divergence, distance_threshold
    
    def __decay_epsilon(self):
        self.__epsilon = max(self.__epsilon * self.params.epsilon_decay, self.params.epsilon_min)


    def __create_empty_states(self):
        states = [self.env.create_empty_state(index=float(i)) for i in range(self.params.batch_size)]
        return tf.convert_to_tensor(states)


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
        critic_optimizer = tf.keras.optimizers.Nadam(self.params.critic_learning_rate, clipvalue=0.5, clipnorm=1.0)
        actor_optimizer = tf.keras.optimizers.Nadam(self.params.actor_learning_rate, clipvalue=0.5, clipnorm=1.0, decay=0.001)

        total_episodes = 500

        total_exploration_steps = math.ceil(200000 / self.params.batch_size) * self.params.batch_size
        total_demonstration_steps = 100 #math.ceil(len(self.encoded_payloads) / self.params.batch_size) * self.params.batch_size * 2

        run_demonstrations = run_demonstrations and self.encoded_payloads is not None

        buffer_size = total_exploration_steps + total_demonstration_steps if run_demonstrations else total_exploration_steps

        self.params.buffer_size = buffer_size

        buffer = ReplayBuffer(
            state_size=self.env.state_size,
            embedding_size=self.env.embedding_size,
            action_size=self.env.action_size,
            buffer_capacity=buffer_size,
            batch_size=self.params.batch_size,
            demonstrations_count=total_exploration_steps,
            actor_model=actor_model,
            policy=lambda state, training: self.policy(state, PolicyType.NORMAL.value, training=training),
            target_policy=lambda state, training: self.policy(state, PolicyType.TARGET.value, training=training),
            target_critic=target_critic,
            critic_model=critic_model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=self.params.gamma
        )

        avg_batch_rewards = []
        avg_divergences = []
        
        frame = 0

        reporter = Reporter()

        print('Starting reporter...')
        reporter.start(self.params)

        if run_demonstrations:
            print('Gathering demonstrations...')

        ep = 1

        end_ddpg = False

        if run_demonstrations:
            prev_states = self.__create_empty_states()

            for i in range(0, total_demonstration_steps, self.params.batch_size):
                actions = tf.convert_to_tensor([random.choice(self.encoded_payloads) for _ in range(self.params.batch_size)])

                env_tuples = self.__run_actions(actions, prev_states, buffer, ignore_episode=False, is_demonstration=True)
                prev_states = tf.convert_to_tensor([env_tuple[0] for env_tuple in env_tuples])
                
                done = True in [env_tuple[2] for env_tuple in env_tuples]

                if done:
                    prev_states = self.__create_empty_states()

                print(f'{i + self.params.batch_size}/{total_demonstration_steps} demonstration observations gathered.')

            print('Transitions gathered.')

        for ep in range(1, total_episodes + 1):
            prev_states = self.__create_empty_states()

            # Update perturbed actor at beginning of episode for stability.
            # (Pg. 3 of Adapative Parameter Space Noise paper).
            self.__update_perturbed_actor()

            while not end_ddpg:
                prev_stddev = self.__stddev

                interactions = self.__get_perturbed_actions(prev_states)
                actions = interactions[0]

                env_tuples = self.__run_actions(actions, prev_states, buffer, ignore_episode=False, is_demonstration=False)
                states = tf.convert_to_tensor([env_tuple[0] for env_tuple in env_tuples])
                
                done = True in [env_tuple[2] for env_tuple in env_tuples]

                frame += self.params.batch_size

                avg_batch_reward = sum([env_tuple[1] for env_tuple in env_tuples]) / self.params.batch_size
                avg_batch_rewards.append(avg_batch_reward)

                avg_reward = np.mean(avg_batch_rewards)

                divergence = interactions[1]

                avg_divergences.append(divergence)
                avg_divergence = np.mean(avg_divergences)

                distance_threshold = interactions[2]

                running_stat = DDPGRunningStatistic(
                    epsiode=ep,
                    frame=frame,
                    total_avg_reward=avg_reward,
                    is_demonstration=False,
                    stddev=prev_stddev,
                    epsilon=self.__epsilon,
                    total_avg_kl_divergence=avg_divergence,
                    distance_threshold=distance_threshold
                )
                
                reporter.record_running_statistic(running_stat)

                payload_stats = [
                    DDPGPayloadStatistic(
                        epsiode=ep,
                        frame=frame,
                        payload=self.env.get_payload(actions[i]),
                        reward=env_tuples[i][1],
                        is_demonstration=False
                    ) for i in range(len(actions)) if env_tuples[i][1] > 0.0
                ]
            
                for stat in payload_stats:
                    reporter.record_payload_statistic(stat)

                if not self.params.constant_stddev:
                    self.__decay_epsilon()

                if frame > total_exploration_steps:
                    done = True
                    end_ddpg = True
                
                buffer.learn()
                self.update_target(target_actor.variables, actor_model.variables, self.params.tau)
                self.update_target(target_critic.variables, critic_model.variables, self.params.tau)

                # End this episode when `done` is True
                if done:
                    break

                prev_states = states

            if end_ddpg:
                break

            print("[{}] Episode: {}, Avg Reward: {}, Total Frame Count: {}".format(datetime.datetime.now(), ep, 'N/A' if avg_reward == None else avg_reward, frame))
