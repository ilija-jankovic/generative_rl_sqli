import math
from typing import List
import tensorflow as tf

from .enums.policy_type import PolicyType
    
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.0001
L2_WEIGHT = 0.0001

ACTOR_LSTM_UNITS = 512
CRITIC_LSTM_UNITS = 512

class PPOActorCritic:
    dictionary_length: int
    action_size: int
    state_size: int
    embedding_size: int
    batch_size: int
    embeddings: List[List[int]]

    actor_model: tf.keras.Model
    actor_model_old: tf.keras.Model
    critic_model: tf.keras.Model

    actor_optimizer: tf.compat.v1.mixed_precision.MixedPrecisionLossScaleOptimizer
    critic_optimizer: tf.compat.v1.mixed_precision.MixedPrecisionLossScaleOptimizer

    def update_old_actor_weights(self):
        self.actor_model_old.set_weights(self.actor_model.get_weights())

    def __init_models(self):
        # Strategy to utilise multiple GPUs.
        #
        # HierarchicalCopyAllReduce for multi-GPU setup on single machine recommendation from:
        # https://github.com/y33-j3T/Coursera-Deep-Learning/blob/master/Custom%20and%20Distributed%20Training%20with%20Tensorflow/Week%204%20-%20Distributed%20Training/C2_W4_Lab_2_multi-GPU-mirrored-strategy.ipynb
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

        device_count = strategy.num_replicas_in_sync
        print('Number of devices: {}'.format(device_count))

        with strategy.scope():
            self.actor_model = self.get_actor(device_count)
            self.actor_model_old = self.get_actor(device_count)
            self.critic_model = self.get_critic(device_count)

            self.actor_optimizer = tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(
                tf.keras.optimizers.Nadam(
                    ACTOR_LEARNING_RATE,
                    clipvalue=0.5,
                    clipnorm=1.0,
                    beta_1=0.999,
                    beta_2=0.999,
                    decay=0.001
                ))

            # Mixed precision gives significant performance increase:
            # https://developer.nvidia.com/automatic-mixed-precision
            #
            # Nadam for RNNs recommended by OverLordGoldDragon:
            # https://stackoverflow.com/questions/48714407/rnn-regularization-which-component-to-regularize/58868383#58868383
            #
            # Optimisers inside strategy:
            # https://www.tensorflow.org/tutorials/distribute/custom_training#training_loop
            self.critic_optimizer = tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(
                tf.keras.optimizers.Nadam(
                    CRITIC_LEARNING_RATE,
                    clipvalue=0.5,
                    clipnorm=1.0,
                    beta_1=0.999,
                    beta_2=0.999
                ))
            
            self.update_old_actor_weights()

    def __init__(
        self,
        dictionary_length: int,
        action_size: int,
        state_size: int,
        embedding_size: int,
        batch_size: int,
        embeddings: List[List[int]]
    ):
        assert(dictionary_length > 0)
        assert(embedding_size > 0)
        assert(batch_size > 0)

        self.dictionary_length = dictionary_length
        self.action_size = action_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.embeddings = embeddings

        self.__init_models()

    @tf.function
    def get_embeddings_from_probabilities(self, probabilities):
        # Log probabilities as tf.random.categorical expects log probabilities.
        probabilities = tf.math.log(probabilities)

        indices = tf.random.categorical(probabilities, num_samples=1, dtype=tf.int32)
        indices = tf.squeeze(indices)

        embeddings = tf.convert_to_tensor(self.embeddings, dtype=tf.float32)
        
        return indices, tf.gather(embeddings, indices)

    def __create_lstm_layer(self, units: int, return_tensors: bool = True):
        return tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units,
                return_state=return_tensors,
                return_sequences=return_tensors,
                kernel_initializer=tf.keras.initializers.Orthogonal(),
                kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
                kernel_constraint=tf.keras.constraints.max_norm(3),
                recurrent_constraint=tf.keras.constraints.max_norm(3),
                bias_constraint=tf.keras.constraints.max_norm(3)
            ))

    def __create_hidden_dense_layer(self, units: int):
        return tf.keras.layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
            kernel_constraint=tf.keras.constraints.max_norm(3),
            bias_constraint=tf.keras.constraints.max_norm(3)
        )

    def get_actor(self, device_count: int):
        c_padding = ACTOR_LSTM_UNITS % self.embedding_size
        c_size = math.ceil(ACTOR_LSTM_UNITS / self.embedding_size)

        # Input = RL state size + LSTM cell state size + single input for average embedding across action fragment.
        input_size = self.state_size + c_size + 1

        batch_size = self.batch_size * device_count

        input_lstm = tf.keras.layers.Input(shape=(input_size, self.embedding_size), batch_size=batch_size)

        # Add normalisation tf.keras.layers between perturbed tf.keras.layers (pg. 3).
        lstm = self.__create_lstm_layer(ACTOR_LSTM_UNITS)(input_lstm)
        lstm = list(map(lambda state: tf.keras.layers.BatchNormalization()(state), lstm))
        lstm = self.__create_lstm_layer(ACTOR_LSTM_UNITS)(lstm)
        lstm = list(map(lambda state: tf.keras.layers.BatchNormalization()(state), lstm))
        lstm = self.__create_lstm_layer(ACTOR_LSTM_UNITS)(lstm)
        lstm = list(map(lambda state: tf.keras.layers.BatchNormalization()(state), lstm))

        # Output of LSTM guide by Jason Brownlee from:
        # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
        state_h = lstm[1]
        state_c = lstm[2]

        dense = self.__create_hidden_dense_layer(1024)(state_h)
        dense = tf.keras.layers.BatchNormalization()(dense)
        dense = self.__create_hidden_dense_layer(1024)(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)
        dense = tf.keras.layers.Dense(self.dictionary_length, activation='softmax')(dense)

        padded_state_c_output = tf.keras.layers.Lambda(lambda state_c: tf.pad(state_c, [[0, 0], [0, c_padding]]))(state_c)
        indices_output, embedding_output = tf.keras.layers.Lambda(self.get_embeddings_from_probabilities)((dense))

        return tf.keras.Model([input_lstm], [padded_state_c_output, indices_output, embedding_output])

    def get_critic(self, device_count: int):
        batch_size = self.batch_size * device_count

        state_input = tf.keras.layers.Input(shape=(self.state_size, self.embedding_size), batch_size=batch_size)

        lstm_state = self.__create_lstm_layer(CRITIC_LSTM_UNITS)(state_input)
        lstm_state = self.__create_lstm_layer(CRITIC_LSTM_UNITS)(lstm_state)
        lstm_state = self.__create_lstm_layer(CRITIC_LSTM_UNITS, return_tensors=False)(lstm_state)

        dense = self.__create_hidden_dense_layer(1024)(lstm_state)
        dense = self.__create_hidden_dense_layer(1024)(dense)
        dense = self.__create_hidden_dense_layer(512)(dense)
        dense = self.__create_hidden_dense_layer(256)(dense)
        dense = self.__create_hidden_dense_layer(128)(dense)
        dense = self.__create_hidden_dense_layer(64)(dense)
        dense = self.__create_hidden_dense_layer(32)(dense)
        dense = self.__create_hidden_dense_layer(16)(dense)
        outputs = tf.keras.layers.Dense(1)(dense)

        return tf.keras.Model([state_input], outputs)

    @tf.function
    def get_embedded_lstm_input(self, rl_states, embeddings, lstm_states):
        embeddings = tf.reshape(embeddings, [self.batch_size, -1, self.embedding_size])
        lstm_states = tf.reshape(lstm_states, [self.batch_size, -1, self.embedding_size])

        input = tf.concat([rl_states, embeddings, lstm_states], axis=1)

        return tf.reshape(input, [self.batch_size, -1, self.embedding_size])

    @tf.function
    def concat_next_token_indicies(self, actions, action_index, action_index_float, embeddings, type: int, training: bool, rl_states, lstm_states):
        batch_size = tf.convert_to_tensor(self.batch_size, dtype=tf.int32)

        input = self.get_embedded_lstm_input(rl_states, embeddings, lstm_states)

        normal_policy_id = tf.constant(PolicyType.NORMAL.value, dtype=tf.int32)

        output = tf.cond(
            tf.equal(type, normal_policy_id),
                true_fn=lambda: self.actor_model(input, training=training),
                false_fn=lambda: self.actor_model_old(input, training=training))

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
        batch_size = tf.constant(self.batch_size, dtype=tf.int32)
        action_size = tf.constant(self.action_size, dtype=tf.int32)
        dictionary_length = tf.constant(self.dictionary_length - 1, dtype=tf.int32)

        actions = tf.fill([batch_size, action_size], dictionary_length)
        
        embeddings = tf.zeros([batch_size, self.embedding_size], dtype=tf.float32)
        lstm_states = tf.zeros([batch_size, ACTOR_LSTM_UNITS + ACTOR_LSTM_UNITS % self.embedding_size], dtype=tf.float32)

        action_index = tf.constant(0, dtype=tf.int32)
        action_index_float = tf.constant(0.0, dtype=tf.float32)

        actions, *_ = tf.while_loop(
            cond=lambda *_: True,
            body=self.concat_next_token_indicies,
            loop_vars=(actions, action_index, action_index_float, embeddings, type, training, states, lstm_states),
            maximum_iterations=action_size,
        )

        return actions