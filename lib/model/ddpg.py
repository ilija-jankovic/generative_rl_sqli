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
    
    def __init__(self, env: Environment, encoded_payloads: List[List[int]], params: DDPGHyperparameters, actor_lstm_units: int = 256):
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
    def mask_probabilities(self, probabilities, action: tf.Tensor):
        payload = tf.py_function(self.env.get_payload, [action], tf.string)

        # Avoid unnecessary mask construction if psi is zero, as the mask will always
        # be ones in this case.
        return tf.cond(tf.less_equal(self.params.psi, 0.0),
            true_fn=lambda: probabilities,
            false_fn=lambda: probabilities * self.get_mask(payload))
    
    @tf.function
    def penalise_non_ast(self, probabilities_batch, actions):
        probabilities_batch = [self.mask_probabilities(probabilities_batch[i], actions[i]) for i in range(self.params.batch_size)]

        return tf.convert_to_tensor(probabilities_batch, dtype=tf.float32)
    
    # Modified solution by chasep255 from: 
    # https://stackoverflow.com/questions/37246030/how-to-change-the-temperature-of-a-softmax-output-in-keras
    @tf.function
    def calculate_temperatured_probabilities(self, softmax_probabilities):
        '''
        `probabilities` expected to be calculated from last softmax layer of
        a neural network.
        '''
        logits = tf.math.log(softmax_probabilities) / self.params.temperature

        return tf.math.exp(logits) / tf.reduce_sum(tf.math.exp(logits))
    
    def get_embeddings_from_probabilities(self, probabilities, actions):
        probabilities = self.calculate_temperatured_probabilities(probabilities)
        probabilities = self.penalise_non_ast(probabilities, actions)

        # Log probabilities as tf.random.categorical expects log probabilities.
        probabilities = tf.math.log(probabilities)

        indices = tf.random.categorical(probabilities, num_samples=1, dtype=tf.int32)
        indices = tf.squeeze(indices)

        embeddings = tf.convert_to_tensor(self.env.embeddings, dtype=tf.float32)
        
        return indices, tf.gather(embeddings, indices)

    def get_actor(self):
        C_PADDING = self.actor_lstm_units % self.env.embedding_size

        dictionary_length = len(self.env.dictionary)

        input_lstm = layers.Input(shape=(None, self.env.embedding_size), batch_size=self.params.batch_size)
        input_actions = layers.Input(shape=(self.env.action_size), batch_size=self.params.batch_size, dtype=tf.int32)

        # Add normalisation layers between perturbed layers (pg. 3).
        lstm = layers.Bidirectional(layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True, kernel_initializer=tf.keras.initializers.Orthogonal()))(input_lstm)
        lstm = list(map(lambda state: layers.BatchNormalization()(state), lstm))
        lstm = layers.Bidirectional(layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True, kernel_initializer=tf.keras.initializers.Orthogonal()))(lstm)
        lstm = list(map(lambda state: layers.BatchNormalization()(state), lstm))
        lstm = layers.Bidirectional(layers.CuDNNLSTM(self.actor_lstm_units, return_state=True, return_sequences=True, kernel_initializer=tf.keras.initializers.Orthogonal()))(lstm)
        lstm = list(map(lambda state: layers.BatchNormalization()(state), lstm))

        # Output of LSTM guide by Jason Brownlee from:
        # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
        state_h = lstm[1]
        state_c = lstm[2]

        dense = layers.Dense(1024, activation='relu')(state_h)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dense(1024, activation='relu')(dense)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dense(dictionary_length, activation='softmax')(dense)

        padded_state_c_output = layers.Lambda(lambda state_c: tf.pad(state_c, [[0, 0], [0, C_PADDING]]))(state_c)
        indices_output, embedding_output = layers.Lambda(lambda output: self.get_embeddings_from_probabilities(output[0], output[1]))((dense, input_actions))

        return keras.Model([input_lstm, input_actions], [padded_state_c_output, indices_output, embedding_output])

    def get_critic(self):
        LSTM_UNITS = 256

        state_input = layers.Input(shape=(self.env.state_size, self.params.embedding_size), batch_size=self.params.batch_size)

        lstm_state = layers.Bidirectional(layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True))(state_input)
        lstm_state = layers.Bidirectional(layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True))(lstm_state)
        lstm_state = layers.Bidirectional(layers.CuDNNLSTM(LSTM_UNITS))(lstm_state)

        action_input = layers.Input(shape=(self.env.action_size,), batch_size=self.params.batch_size, dtype=tf.int32)
        embeddding_input = layers.Lambda(lambda action: tf.gather(self.env.embeddings, action))(action_input)

        lstm_action = layers.Bidirectional(layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True))(embeddding_input)
        lstm_action = layers.Bidirectional(layers.CuDNNLSTM(LSTM_UNITS, return_state=True, return_sequences=True))(lstm_action)
        lstm_action = layers.Bidirectional(layers.CuDNNLSTM(LSTM_UNITS))(lstm_action)

        concat = layers.Concatenate()([lstm_state, lstm_action])

        dense = layers.Dense(1024, activation="relu")(concat)
        dense = layers.Dense(1024, activation="relu")(dense)
        dense = layers.Dense(512, activation="relu")(dense)
        dense = layers.Dense(256, activation="relu")(dense)
        dense = layers.Dense(128, activation="relu")(dense)
        dense = layers.Dense(64, activation="relu")(dense)
        dense = layers.Dense(32, activation="relu")(dense)
        dense = layers.Dense(16, activation="relu")(dense)
        outputs = layers.Dense(1)(dense)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def get_embedded_lstm_input(self, rl_states, embeddings, lstm_states):
        embeddings = tf.reshape(embeddings, [self.params.batch_size, -1, self.env.embedding_size])
        lstm_states = tf.reshape(lstm_states, [self.params.batch_size, -1, self.env.embedding_size])

        input = tf.concat([rl_states, embeddings, lstm_states], axis=1)

        return tf.reshape(input, [self.params.batch_size, -1, self.env.embedding_size])

    def concat_next_token_indicies(self, actions, action_index, action_index_float, embeddings, type: int, training: bool, rl_states, lstm_states):
        batch_size = self.params.batch_size

        input = (self.get_embedded_lstm_input(rl_states, embeddings, lstm_states), actions)

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
        env_tuples = [self.env.perform_action(actions[i], batch_index=i, ignore_episode=ignore_episode) for i in range(len(actions))]

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
        states = [self.env.create_empty_state(index=i) for i in range(self.params.batch_size)]

        return tf.convert_to_tensor(states)
    

    def __calculate_mean_kl_divergence(self, states_batch: tf.Tensor, perturbed_actions_batch: tf.Tensor):
        steps = len(states_batch)

        assert(steps > 0 and len(perturbed_actions_batch) == steps)

        divergences = []

        for i in range(steps):
            states = states_batch[i]
            perturbed_actions = perturbed_actions_batch[i]

            actions = self.policy(states, PolicyType.NORMAL.value, training=False)

            divergences.append(self.get_kl_divergence(actions, perturbed_actions))

        return tf.reduce_mean(divergences)


    def __learn(self, buffer: ReplayBuffer, reporter: Reporter, episode: int, frame: int, n_step_rollout = 5):
        batch_indices, chosen_probabilities = buffer.sample_indices()

        state_batch = buffer.state_buffer[batch_indices]
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)

        state_batches, action_batches, reward_batches, stats, done = self.env.perform_n_step_rollout(
            policy=lambda state, training: self.policy(state, PolicyType.NORMAL.value, training=training),
            perturbed_policy=lambda state, training: self.policy(state, PolicyType.PERTURBED.value, training=training),
            state_batch=state_batch,
            episode=episode,
            frame=frame,
            n=n_step_rollout
        )

        average_reward = tf.reduce_mean(reward_batches)
        divergence = self.__calculate_mean_kl_divergence(state_batches[:-1], action_batches)

        for stat in stats:
            reporter.record_payload_statistic(stat)

        for i in range(n_step_rollout):
            observations = [(state_batches[i][j], action_batches[i][j], reward_batches[i][j], state_batches[i + 1][j]) for j in range(self.params.batch_size)]

            buffer.record(observations, is_demonstration=False)

        critic_loss, actor_loss = buffer.learn(
            batch_indices=batch_indices,
            chosen_probabilities=chosen_probabilities,
            n_step_rollout=n_step_rollout,
            reward_batches=reward_batches,
            last_state_batch=state_batches[-2],
            last_action_batch=action_batches[-1]
        )

        self.update_target(self.target_actor.variables, self.actor_model.variables, self.params.tau)
        self.update_target(self.target_critic.variables, self.critic_model.variables, self.params.tau)

        return average_reward, divergence, done, critic_loss, actor_loss


    def run(self, run_demonstrations: bool):
        actor_model = self.get_actor()
        actor_perturbed = self.get_actor()
        critic_model = self.get_critic()

        target_actor = self.get_actor()
        target_critic = self.get_critic()

        self.actor_model = actor_model
        self.actor_perturbed = actor_perturbed
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.critic_model = critic_model

        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

        # Nadam for RNNs recommended by OverLordGoldDragon:
        # https://stackoverflow.com/questions/48714407/rnn-regularization-which-component-to-regularize/58868383#58868383
        critic_optimizer = tf.keras.optimizers.Nadam(self.params.critic_learning_rate, clipvalue=0.5, clipnorm=1.0)
        actor_optimizer = tf.keras.optimizers.Nadam(self.params.actor_learning_rate, clipvalue=0.5, clipnorm=1.0, decay=0.001)

        total_exploration_steps = math.ceil(200000 / self.params.batch_size) * self.params.batch_size
        total_demonstration_steps = math.ceil(len(self.encoded_payloads) / self.params.batch_size) * self.params.batch_size * 2 if run_demonstrations else 0

        run_demonstrations = run_demonstrations and self.encoded_payloads is not None

        buffer_size = total_exploration_steps + total_demonstration_steps if run_demonstrations else total_exploration_steps

        self.params.buffer_size = buffer_size

        buffer = ReplayBuffer(
            state_size=self.env.state_size,
            embedding_size=self.env.embedding_size,
            action_size=self.env.action_size,
            buffer_capacity=buffer_size,
            batch_size=self.params.batch_size,
            demonstrations_count=total_demonstration_steps,
            actor_model=actor_model,
            policy=lambda state, training: self.policy(state, PolicyType.NORMAL.value, training=training),
            target_policy=lambda state, training: self.policy(state, PolicyType.TARGET.value, training=training),
            target_critic=target_critic,
            critic_model=critic_model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=self.params.gamma,
            rollout_weight=self.params.rollout_weight,
            l2_weight=self.params.l2_weight,
            priority_weight=self.params.priority_weight
        )
        
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
        else:
            states = self.__create_empty_states()

            interactions = self.__get_perturbed_actions(states)
            actions = interactions[0]

            self.__run_actions(actions, states, buffer, ignore_episode=True, is_demonstration=False)


        while True:
            states = self.__create_empty_states()

            # Update perturbed actor at beginning of episode for stability.
            # (Pg. 3 of Adapative Parameter Space Noise paper).
            self.__update_perturbed_actor()

            while not end_ddpg:
                prev_stddev = self.__stddev

                avg_reward, divergence, done, critic_loss, actor_loss = self.__learn(buffer, reporter, episode=ep, frame=frame)

                running_stat = DDPGRunningStatistic(
                    epsiode=ep,
                    frame=frame,
                    avg_n_step_reward=avg_reward,
                    is_demonstration=False,
                    stddev=prev_stddev,
                    epsilon=self.__epsilon,
                    avg_n_step_kl_divergence=divergence,
                    distance_threshold=None,
                    critic_loss=critic_loss,
                    actor_loss=actor_loss
                )
                
                reporter.record_running_statistic(running_stat)

                if not self.params.constant_stddev:
                    self.__decay_epsilon()
                
                frame += self.params.batch_size * self.params.n_step_rollout

                if frame > total_exploration_steps:
                    done = True
                    end_ddpg = True

                print("[{}] Episode: {}, Total Frame Count: {}, Average Batch Reward: {}".format(datetime.datetime.now(), ep, frame, avg_reward))

                # End this episode when `done` is True
                if done:
                    ep += 1
                    break

                prev_states = states

            if end_ddpg:
                break
