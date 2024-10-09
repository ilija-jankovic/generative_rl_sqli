import os
from typing import List

from lib.pretrain_actor_type import PretrainActorType

from ..hyperparameters import ACTOR_DENSE_UNITS, INITIAL_ACTOR_LEARNING_RATE, ACTOR_LSTM_UNITS, \
    INITIAL_CRITIC_LEARNING_RATE, L2_WEIGHT, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, \
    LR_SCHEDULE_DECAY_RATE, LR_SCHEDULE_DECAY_STEPS, PRETRAIN_ACTOR_TYPE, PRETRAINING_LEARNING_RATE

# Sets TF logger level to ERROR.
#
# Important to place before TF import, as stated by Matt Haythornthwaite
# from: https://stackoverflow.com/a/64448286
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from .policy_type import PolicyType

# Strategy to utilise multiple GPUs.
#
# HierarchicalCopyAllReduce for multi-GPU setup on single machine recommendation from:
# https://github.com/y33-j3T/Coursera-Deep-Learning/blob/master/Custom%20and%20Distributed%20Training%20with%20Tensorflow/Week%204%20-%20Distributed%20Training/C2_W4_Lab_2_multi-GPU-mirrored-strategy.ipynb
#strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

#device_count = strategy.num_replicas_in_sync


class PPOActorCritic:
    dictionary_length: int
    action_size: int
    state_size: int
    embedding_size: int
    embeddings: List[List[int]]

    actor_model: tf.keras.Model
    actor_model_old: tf.keras.Model
    critic_model: tf.keras.Model

    actor_pretraining_optimizer: tf.keras.mixed_precision.LossScaleOptimizer

    actor_optimizer: tf.keras.mixed_precision.LossScaleOptimizer
    critic_optimizer: tf.keras.mixed_precision.LossScaleOptimizer
    
    
    def __create_pretraining_optimizer(self):
        return tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Nadam(learning_rate=PRETRAINING_LEARNING_RATE),
        )
    
    # Mixed precision gives significant performance increase:
    # https://developer.nvidia.com/automatic-mixed-precision
    #
    # Parameter values are based on section "13 core implementation details" from
    # the ICLR PPO research guide:
    # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    #
    # Nadam for RNNs recommended by OverLordGoldDragon:
    # https://stackoverflow.com/questions/48714407/rnn-regularization-which-component-to-regularize/58868383#58868383
    def __create_rl_optimizer(self, initial_learning_rate: float):
        assert(initial_learning_rate > 0.0)
        
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=LR_SCHEDULE_DECAY_STEPS,
            decay_rate=LR_SCHEDULE_DECAY_RATE,
        )

        return tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Nadam(
                learning_rate=learning_rate_schedule,
                beta_1=ADAM_BETA1,
                beta_2=ADAM_BETA2,
                epsilon=ADAM_EPSILON,
            ))

    def __init_models(self):
        tf.keras.backend.set_floatx('float64')

        if PRETRAIN_ACTOR_TYPE == PretrainActorType.LOAD_PRETRAINED:
            dirname = os.path.dirname(__file__)

            # Allow Lambda layer loading.
            tf.keras.config.enable_unsafe_deserialization()
            
            self.actor_model = tf.keras.models.load_model(f'{dirname}/../../pretrained_actor.keras')
            self.actor_model_old = tf.keras.models.load_model(f'{dirname}/../../pretrained_actor.keras')
        else:
            self.actor_model = self.get_actor(name='ACTOR')
            self.actor_model_old = self.get_actor(name='ACTOR_OLD')
        
        self.critic_model = self.get_critic(name='CRITIC')

        self.actor_model.summary()
        self.critic_model.summary()

        self.actor_pretraining_optimizer = self.__create_pretraining_optimizer()
        
        self.actor_optimizer = self.__create_rl_optimizer(INITIAL_ACTOR_LEARNING_RATE)
        self.critic_optimizer = self.__create_rl_optimizer(INITIAL_CRITIC_LEARNING_RATE)
        
        self.update_old_actor_weights()

    def __init__(
        self,
        dictionary_length: int,
        action_size: int,
        state_size: int,
        embedding_size: int,
        embeddings: List[List[int]]
    ):
        assert(dictionary_length > 0)
        assert(embedding_size > 0)

        self.dictionary_length = dictionary_length
        self.action_size = action_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.embeddings = embeddings

        self.__init_models()

    def update_old_actor_weights(self):
        self.actor_model_old.set_weights(self.actor_model.get_weights())

    @tf.function
    def get_embeddings_from_probabilities(
        self,
        probabilities: tf.Tensor,
        chosen_indices: tf.Tensor,
        use_chosen_indices: tf.Tensor,
    ):
        chosen_indices = tf.cond(
            tf.equal(use_chosen_indices, True),
            true_fn=lambda: chosen_indices,
        
            # Log probabilities as tf.random.categorical expects log probabilities.
            false_fn=lambda: tf.random.categorical(tf.math.log(probabilities), num_samples=1, dtype=tf.int32),
        )
        chosen_indices = tf.squeeze(chosen_indices)
        
        # TODO: Consider using tf.gather_nd for performance increase:
        # https://github.com/matterport/Mask_RCNN/issues/749#issuecomment-1497595166
        chosen_embeddings = tf.gather(self.embeddings, chosen_indices)
        chosen_embeddings = tf.cast(chosen_embeddings, dtype=tf.float64)

        chosen_probabilities = tf.gather(probabilities, chosen_indices, axis=1, batch_dims=1)
        
        return chosen_indices, chosen_embeddings, chosen_probabilities

    def __create_lstm(self, units: int):
        return tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            unroll=True,
            activation='tanh',
            
            # Initialise LSTMs with normal distribution (stddev=1.0) and biases of zero,
            # as stated under section "5 LSTM implementation details" from the ICLR PPO 
            # research guide:
            # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            bias_initializer='zeros',

            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
            kernel_constraint=tf.keras.constraints.max_norm(3),
            bias_constraint=tf.keras.constraints.max_norm(3),
        )

    def __create_hidden_dense_layer(self, units: int, activation: str):
        return tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
            kernel_constraint=tf.keras.constraints.max_norm(3),
            bias_constraint=tf.keras.constraints.max_norm(3),
        )

    def get_actor(self, name: str):
        input_rl_state = tf.keras.layers.Input(shape=[self.state_size,])

        dense_rl_state = self.__create_hidden_dense_layer(
            ACTOR_DENSE_UNITS, activation='tanh',
        )(input_rl_state)
        dense_rl_state = tf.keras.layers.BatchNormalization()(dense_rl_state)

        dense_rl_state = self.__create_hidden_dense_layer(
            ACTOR_DENSE_UNITS, activation='tanh',
        )(dense_rl_state)
        dense_rl_state = tf.keras.layers.BatchNormalization()(dense_rl_state)

        dense_rl_state = self.__create_hidden_dense_layer(
            ACTOR_DENSE_UNITS, activation='tanh',
        )(dense_rl_state)
        dense_rl_state = tf.keras.layers.BatchNormalization()(dense_rl_state)

        input_state_h = tf.keras.layers.Input(shape=[ACTOR_LSTM_UNITS,])
        input_state_c = tf.keras.layers.Input(shape=[ACTOR_LSTM_UNITS,])

        input_lstm_state = (
            input_state_h,
            input_state_c,
        )

        # Average embedding input.
        input_embedding = tf.keras.layers.Input(shape=[1, self.embedding_size,],)

        lstm, *lstm_state = self.__create_lstm(
            ACTOR_LSTM_UNITS,
        )(
            input_embedding,
            initial_state=input_lstm_state,
        )

        lstm, *lstm_state = self.__create_lstm(
            ACTOR_LSTM_UNITS,
        )(
            lstm,
            initial_state=lstm_state,
        )
        
        lstm, *lstm_state = self.__create_lstm(
            ACTOR_LSTM_UNITS,
        )(
            lstm,
            initial_state=lstm_state,
        )

        lstm = tf.keras.layers.Flatten()(lstm)

        concat = tf.keras.layers.Concatenate()([dense_rl_state, lstm,])

        dense = self.__create_hidden_dense_layer(ACTOR_DENSE_UNITS, activation='tanh')(concat)
        dense = tf.keras.layers.BatchNormalization()(dense)
        
        dense = self.__create_hidden_dense_layer(ACTOR_DENSE_UNITS, activation='tanh')(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)

        dense = tf.keras.layers.Lambda(lambda dense: dense / 2.0, output_shape=(ACTOR_DENSE_UNITS,))(dense)
        dense = tf.keras.layers.Dense(self.dictionary_length, activation='softmax')(dense)

        return tf.keras.Model(
            inputs=[
                input_rl_state,
                *input_lstm_state,
                input_embedding,
            ],
            outputs=[
                dense,
                *lstm_state,
            ],
            name=name,
        )
        
    def get_critic(self, name: str):
        input_rl_state = tf.keras.layers.Input(shape=[self.state_size,])

        dense = self.__create_hidden_dense_layer(
            512,
            activation='relu',
        )(input_rl_state)
        dense = tf.keras.layers.BatchNormalization()(dense)
        
        dense = self.__create_hidden_dense_layer(
            512,
            activation='relu',
        )(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)
        
        dense = self.__create_hidden_dense_layer(
            256,
            activation='relu',
        )(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)
        
        dense = self.__create_hidden_dense_layer(
            128,
            activation='relu',
        )(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)
        
        dense = self.__create_hidden_dense_layer(
            64,
            activation='relu',
        )(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)
        
        dense = self.__create_hidden_dense_layer(
            32,
            activation='relu',
        )(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)

        dense = self.__create_hidden_dense_layer(
            16,
            activation='relu',
        )(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)
        
        dense = self.__create_hidden_dense_layer(
            8,
            activation='relu',
        )(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)
        
        output = tf.keras.layers.Dense(1, activation='linear',)(dense)

        return tf.keras.Model(
            inputs=[input_rl_state,],
            outputs=output,
            name=name,
        )

    @tf.function
    def concat_next_token_indicies(
        self, 
        actions: tf.Tensor,
        probabilities: tf.Tensor,
        batch_size: int,
        action_index: tf.Tensor,
        action_index_float: tf.Tensor,
        lstm_state: tf.Tensor,
        embeddings: tf.Tensor,
        type: int,
        states: tf.Tensor,
        actions_reference: tf.Tensor,
        use_actions_reference: bool,
    ):
        input = (states, *lstm_state, embeddings,)

        normal_policy_id = tf.constant(PolicyType.NORMAL.value, dtype=tf.int32)

        one_hot_probabilities, *lstm_state = tf.cond(
            tf.equal(type, normal_policy_id),
                true_fn=lambda: self.actor_model(input, training=True),
                false_fn=lambda: self.actor_model_old(input, training=False))
        
        chosen_indices, chosen_embeddings, chosen_probabilities = tf.cond(
            tf.equal(use_actions_reference, True),
            true_fn=lambda: self.get_embeddings_from_probabilities(one_hot_probabilities, actions_reference[:,action_index], True),
            false_fn=lambda: self.get_embeddings_from_probabilities(one_hot_probabilities, tf.fill([batch_size, 1], -1), False)
        )

        # action_index_float is the length of the action after incrementing, which
        # is then used in the below embedding average calculation.
        action_index_float = tf.add(action_index_float, 1.0)

        chosen_embeddings = tf.reshape(chosen_embeddings, [batch_size, 1, self.embedding_size])

        # Adding to an average solution by Damien and Dan Dascalescu from:
        # https://math.stackexchange.com/questions/22348/how-to-add-and-subtract-values-from-an-average
        embeddings = embeddings + tf.math.divide(chosen_embeddings - embeddings, action_index_float)

        action_indices = tf.range(0, batch_size, dtype=tf.int32)
        action_indices = tf.expand_dims(action_indices, axis=1)
        action_indices = tf.pad(action_indices, [[0, 0], [0, 1]], constant_values=action_index)

        actions = tf.tensor_scatter_nd_update(actions, action_indices, chosen_indices)
        probabilities = tf.tensor_scatter_nd_update(probabilities, action_indices, chosen_probabilities)

        action_index =  tf.add(action_index, 1)
        
        return actions, probabilities, batch_size, action_index, action_index_float, lstm_state, embeddings, type, states, actions_reference, use_actions_reference

    @tf.function
    def policy(
        self,
        states: tf.Tensor,
        type: int,
        batch_size: int,
        actions_reference: tf.Tensor,
        use_actions_reference: bool,
    ):
        '''
        `type` is expected to be the enumerated value of a `PolicyType`.
        
        Setting `use_actions_reference` to `True` ignores action reference.
        This setting chooses tokens stochastically instead of following the reference.
        '''
        action_size = tf.constant(self.action_size, dtype=tf.int32)

        actions = tf.fill([batch_size, action_size], -1)
        probabilities = tf.zeros([batch_size, action_size], dtype=tf.float64)
        
        state_h = tf.zeros([batch_size, ACTOR_LSTM_UNITS,], dtype=tf.float64)
        state_c = tf.zeros([batch_size, ACTOR_LSTM_UNITS,], dtype=tf.float64)

        lstm_state = (
            state_h,
            state_c,
        )
        
        embeddings = tf.zeros([batch_size, 1, self.embedding_size], dtype=tf.float64)

        action_index = tf.constant(0, dtype=tf.int32)
        action_index_float = tf.constant(0.0, dtype=tf.float64)

        actions, probabilities, *_ = tf.while_loop(
            cond=lambda *_: True,
            body=self.concat_next_token_indicies,
            loop_vars=(
                actions,
                probabilities,
                batch_size,
                action_index,
                action_index_float,
                lstm_state,
                embeddings,
                type,
                states,
                actions_reference,
                use_actions_reference,
            ),
            maximum_iterations=action_size,
        )

        action_likelihoods = tf.math.reduce_prod(probabilities, axis=-1, keepdims=True)

        return actions, action_likelihoods
