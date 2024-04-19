# Modification of DDPG Keras example from:
# https://keras.io/examples/rl/ddpg_pendulum/

from contextlib import nullcontext
import datetime
import math
import random
from typing import List
import tensorflow as tf
import numpy as np

from .ddpg_hyperparameters import DDPGHyperparameters
from .ddpg_running_statistic import DDPGRunningStatistic
from .ddpg_payload_statistic import DDPGPayloadStatistic
from .reporter import Reporter

from .enums.policy_type import PolicyType
from .environment import Environment
from .replay_buffer import ReplayBuffer, strategy


class DDPG:
    env: Environment
    encoded_payloads: List[List[int]]
    params: DDPGHyperparameters
    profile: bool
    actor_lstm_units: int
    
    actor_perturbed: tf.keras.Model

    __stddev: float
    
    def __init__(
            self,
            env: Environment,
            encoded_payloads: List[List[int]],
            params: DDPGHyperparameters,
            profile: bool,
            actor_lstm_units: int = 512,
        ):
        assert(params.psi >= 0.0 and params.psi <= 1.0)

        dictionary_length = len(env.dictionary)

        # Ensure last token in dictionary is the empty token.
        assert(dictionary_length > 0 and env.dictionary[-1] == '')

        self.env = env
        self.params = params
        self.profile = profile
        self.actor_lstm_units = actor_lstm_units

        self.__stddev = params.starting_stddev

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
    def get_embeddings_from_probabilities(self, probabilities):
        # Log probabilities as tf.random.categorical expects log probabilities.
        probabilities = tf.math.log(probabilities)

        indices = tf.random.categorical(probabilities, num_samples=1, dtype=tf.int32)
        indices = tf.squeeze(indices)

        embeddings = tf.convert_to_tensor(self.env.embeddings, dtype=tf.float32)
        
        return indices, tf.gather(embeddings, indices)
    
    def __create_lstm_layer(self, units: int, return_tensors: bool = True):
        return tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units,
                return_state=return_tensors,
                return_sequences=return_tensors,
                kernel_initializer=tf.keras.initializers.Orthogonal(),
                kernel_regularizer=tf.keras.regularizers.l2(self.params.l2_weight),
                kernel_constraint=tf.keras.constraints.max_norm(3),
                recurrent_constraint=tf.keras.constraints.max_norm(3),
                bias_constraint=tf.keras.constraints.max_norm(3)
            ))
    
    def __create_hidden_dense_layer(self, units: int):
        return tf.keras.layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.params.l2_weight),
            kernel_constraint=tf.keras.constraints.max_norm(3),
            bias_constraint=tf.keras.constraints.max_norm(3)
        )

    def get_actor(self, device_count: int):
        c_padding = self.actor_lstm_units % self.env.embedding_size
        c_size = math.ceil(self.actor_lstm_units / self.env.embedding_size)

        # Input = RL state size + LSTM cell state size + single input for average embedding across action fragment.
        input_size = self.env.state_size + c_size + 1

        dictionary_length = len(self.env.dictionary)

        batch_size = self.params.batch_size * device_count

        input_lstm = tf.keras.layers.Input(shape=(input_size, self.env.embedding_size), batch_size=batch_size)

        # Add normalisation tf.keras.layers between perturbed tf.keras.layers (pg. 3).
        lstm = self.__create_lstm_layer(self.actor_lstm_units)(input_lstm)
        lstm = list(map(lambda state: tf.keras.layers.BatchNormalization()(state), lstm))
        lstm = self.__create_lstm_layer(self.actor_lstm_units)(lstm)
        lstm = list(map(lambda state: tf.keras.layers.BatchNormalization()(state), lstm))
        lstm = self.__create_lstm_layer(self.actor_lstm_units)(lstm)
        lstm = list(map(lambda state: tf.keras.layers.BatchNormalization()(state), lstm))

        # Output of LSTM guide by Jason Brownlee from:
        # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
        state_h = lstm[1]
        state_c = lstm[2]

        dense = self.__create_hidden_dense_layer(1024)(state_h)
        dense = tf.keras.layers.BatchNormalization()(dense)
        dense = self.__create_hidden_dense_layer(1024)(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)
        dense = tf.keras.layers.Dense(dictionary_length, activation='softmax')(dense)

        padded_state_c_output = tf.keras.layers.Lambda(lambda state_c: tf.pad(state_c, [[0, 0], [0, c_padding]]))(state_c)
        indices_output, embedding_output = tf.keras.layers.Lambda(self.get_embeddings_from_probabilities)((dense))

        return tf.keras.Model([input_lstm], [padded_state_c_output, indices_output, embedding_output])

    def get_critic(self, device_count: int):
        LSTM_UNITS = 512

        batch_size = self.params.batch_size * device_count

        state_input = tf.keras.layers.Input(shape=(self.env.state_size, self.params.embedding_size), batch_size=batch_size)

        lstm_state = self.__create_lstm_layer(LSTM_UNITS)(state_input)
        lstm_state = self.__create_lstm_layer(LSTM_UNITS)(lstm_state)
        lstm_state = self.__create_lstm_layer(LSTM_UNITS, return_tensors=False)(lstm_state)

        action_input = tf.keras.layers.Input(shape=(self.env.action_size,), batch_size=batch_size, dtype=tf.int32)
        embeddding_input = tf.keras.layers.Lambda(lambda action: tf.gather(self.env.embeddings, action))(action_input)

        lstm_action = self.__create_lstm_layer(LSTM_UNITS)(embeddding_input)
        lstm_action = self.__create_lstm_layer(LSTM_UNITS)(lstm_action)
        lstm_action = self.__create_lstm_layer(LSTM_UNITS, return_tensors=False)(lstm_action)

        concat = tf.keras.layers.Concatenate()([lstm_state, lstm_action])

        dense = self.__create_hidden_dense_layer(1024)(concat)
        dense = self.__create_hidden_dense_layer(1024)(dense)
        dense = self.__create_hidden_dense_layer(512)(dense)
        dense = self.__create_hidden_dense_layer(256)(dense)
        dense = self.__create_hidden_dense_layer(128)(dense)
        dense = self.__create_hidden_dense_layer(64)(dense)
        dense = self.__create_hidden_dense_layer(32)(dense)
        dense = self.__create_hidden_dense_layer(16)(dense)
        outputs = tf.keras.layers.Dense(1)(dense)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    @tf.function
    def get_embedded_lstm_input(self, rl_states, embeddings, lstm_states):
        embeddings = tf.reshape(embeddings, [self.params.batch_size, -1, self.env.embedding_size])
        lstm_states = tf.reshape(lstm_states, [self.params.batch_size, -1, self.env.embedding_size])

        input = tf.concat([rl_states, embeddings, lstm_states], axis=1)

        return tf.reshape(input, [self.params.batch_size, -1, self.env.embedding_size])

    @tf.function
    def concat_next_token_indicies(self, actions, action_index, action_index_float, embeddings, type: int, training: bool, rl_states, lstm_states):
        batch_size = tf.convert_to_tensor(self.params.batch_size, dtype=tf.int32)

        input = self.get_embedded_lstm_input(rl_states, embeddings, lstm_states)

        normal_policy_id = tf.constant(PolicyType.NORMAL.value, dtype=tf.int32)
        perturbed_policy_id = tf.constant(PolicyType.PERTURBED.value, dtype=tf.int32)

        output = tf.cond(
            tf.equal(type, normal_policy_id),
                true_fn=lambda: self.actor_model(input, training=training),
                false_fn=lambda: tf.cond(
                tf.equal(type, perturbed_policy_id),
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
        batch_size = tf.constant(self.params.batch_size, dtype=tf.int32)
        action_size = tf.constant(self.env.action_size, dtype=tf.int32)
        dictionary_size = tf.constant(len(self.env.dictionary) - 1, dtype=tf.int32)

        actions = tf.fill([batch_size, action_size], dictionary_size)
        
        embeddings = tf.zeros([batch_size, self.env.embedding_size], dtype=tf.float32)
        lstm_states = tf.zeros([batch_size, self.actor_lstm_units + self.actor_lstm_units % self.env.embedding_size], dtype=tf.float32)

        action_index = tf.constant(0, dtype=tf.int32)
        action_index_float = tf.constant(0.0, dtype=tf.float32)

        actions, *_ = tf.while_loop(
            cond=lambda *_: True,
            body=self.concat_next_token_indicies,
            loop_vars=(actions, action_index, action_index_float, embeddings, type, training, states, lstm_states),
            maximum_iterations=action_size,
        )

        return actions


    def __run_actions(self, actions: tf.Tensor, prev_states: tf.Tensor, buffer: ReplayBuffer, ignore_episode: bool):
        # Recieve state and reward from environment.
        env_tuples = [self.env.perform_action(actions[i], batch_index=i, ignore_episode=ignore_episode) for i in range(len(actions))]

        states = [env_tuple[0] for env_tuple in env_tuples]
        rewards = [env_tuple[1] for env_tuple in env_tuples]
        done_flags = [env_tuple[2] for env_tuple in env_tuples]

        obs_tuples = [(prev_states[i], actions[i], rewards[i], states[i]) for i in range(self.params.batch_size)]

        buffer.record(obs_tuples)

        done = True in done_flags

        return states, rewards, done
    
    
    def __update_perturbed_actor(self):
        self.actor_perturbed.set_weights(self.actor_model.get_weights()) 

        # Adding noise to model weights algorithm by Daan Klijn:
        # https://medium.com/adding-noise-to-network-weights-in-tensorflow/adding-noise-to-network-weights-in-tensorflow-fddc82e851cb
        for layer in self.actor_perturbed.trainable_weights:
            noise = np.random.normal(loc=0.0, scale=self.__stddev, size=layer.shape)
            layer.assign_add(noise)


    def __create_empty_states(self):
        states = [self.env.create_empty_state(index=i) for i in range(self.params.batch_size)]

        return tf.convert_to_tensor(states)


    def __learn(self, buffer: ReplayBuffer, profile: bool):
        critic_losses = []
        actor_losses = []

        if profile:
            tf.profiler.experimental.start('tensorboard_log')

        for learning_step in range(self.params.learnings_per_batch):
            with tf.profiler.experimental.Trace('train', step_num=learning_step, _r=1) if profile else nullcontext():
                critic_loss, actor_loss = buffer.learn()
                
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

            self.update_target(self.target_actor.variables, self.actor_model.variables, self.params.tau)
            self.update_target(self.target_critic.variables, self.critic_model.variables, self.params.tau)

        if profile:
            tf.profiler.experimental.stop()

        avg_critic_loss = tf.reduce_mean(critic_losses)
        avg_actor_loss = tf.reduce_mean(actor_losses)

        return avg_critic_loss, avg_actor_loss
    

    def run(self, run_demonstrations: bool):
        device_count = strategy.num_replicas_in_sync
        print('Number of devices: {}'.format(device_count))

        with strategy.scope():
            actor_model = self.get_actor(device_count)
            actor_perturbed = self.get_actor(device_count)
            critic_model = self.get_critic(device_count)

            target_actor = self.get_actor(device_count)
            target_critic = self.get_critic(device_count)

            # Mixed precision gives significant performance increase:
            # https://developer.nvidia.com/automatic-mixed-precision
            #
            # Nadam for RNNs recommended by OverLordGoldDragon:
            # https://stackoverflow.com/questions/48714407/rnn-regularization-which-component-to-regularize/58868383#58868383
            #
            # Optimisers inside strategy:
            # https://www.tensorflow.org/tutorials/distribute/custom_training#training_loop
            critic_optimizer = tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(
                tf.keras.optimizers.Nadam(
                    self.params.critic_learning_rate,
                    clipvalue=0.5,
                    clipnorm=1.0,
                    beta_1=0.999,
                    beta_2=0.999
                ))
            
            actor_optimizer = tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(
                tf.keras.optimizers.Nadam(
                    self.params.actor_learning_rate,
                    clipvalue=0.5,
                    clipnorm=1.0,
                    beta_1=0.999,
                    beta_2=0.999,
                    decay=0.001
                ))

            self.actor_model = actor_model
            self.actor_perturbed = actor_perturbed
            self.target_actor = target_actor
            self.target_critic = target_critic
            self.critic_model = critic_model

            # Making the weights equal initially
            target_actor.set_weights(actor_model.get_weights())
            target_critic.set_weights(critic_model.get_weights())

            total_exploration_steps = math.ceil(100000 / self.params.batch_size) * self.params.batch_size
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
                policy=lambda state, training: self.policy(state, tf.constant(PolicyType.NORMAL.value, dtype=tf.int32), training=tf.constant(training, dtype=tf.bool)),
                target_policy=lambda state, training: self.policy(state, tf.constant(PolicyType.TARGET.value, dtype=tf.int32), training=tf.constant(training, dtype=tf.bool)),
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

            states = self.__create_empty_states()

            if run_demonstrations:
                for i in range(0, total_demonstration_steps, self.params.batch_size):
                    print(f'{i}/{total_demonstration_steps} demonstration observations gathered.')

                    actions = tf.convert_to_tensor([random.choice(self.encoded_payloads) for _ in range(self.params.batch_size)])

                    states, _, __ = self.__run_actions(actions, states, buffer, ignore_episode=False)

                print('Transitions gathered.')

            while True:
                # Update perturbed actor at beginning of episode for stability.
                # (Pg. 3 of Adapative Parameter Space Noise paper).
                self.__update_perturbed_actor()

                while not end_ddpg:
                    actions = self.policy(states, tf.constant(PolicyType.PERTURBED.value, dtype=tf.int32), training=tf.constant(False, dtype=tf.bool))

                    states, rewards, done = self.__run_actions(actions, states, buffer, ignore_episode=False)
                    
                    avg_main_rollout_reward = float(np.average(rewards))

                    # Don't profile first learning batch as model is initialised and initial casts
                    # are performed by mixed precision optimisers.
                    profile = self.profile and frame != 0
                    critic_loss, actor_loss = self.__learn(buffer, profile=profile)

                    # TODO: Record all successful payloads, even from rollout, as they
                    # are equally valuable to pen-testers.
                    for i in range(self.params.batch_size):
                        reward = rewards[i]

                        if reward > 0.0:
                            action = actions[i]

                            stat = DDPGPayloadStatistic(
                                epsiode=ep,
                                frame=frame,
                                payload=self.env.get_payload(action),
                                reward=reward,
                                is_demonstration=False
                            )

                            reporter.record_payload_statistic(stat)

                    frame += self.params.batch_size

                    running_stat = DDPGRunningStatistic(
                        epsiode=ep,
                        frame=frame,
                        avg_main_rollout_reward=avg_main_rollout_reward,
                        is_demonstration=False,
                        critic_loss=critic_loss,
                        actor_loss=actor_loss
                    )
                    
                    reporter.record_running_statistic(running_stat)

                    if frame > total_exploration_steps:
                        done = True
                        end_ddpg = True

                    print("[{}] Episode: {}, Total Frame Count: {}, Average n-Step Batch Reward: {}".format(datetime.datetime.now(), ep, frame, avg_main_rollout_reward))

                    # End this episode when `done` is True
                    if done:
                        ep += 1
                        break


                if end_ddpg:
                    break
