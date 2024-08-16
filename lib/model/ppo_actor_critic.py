import math
import os
from typing import List

# Sets TF logger level to WARNING.
#
# Important to place before TF import, as stated by Matt Haythornthwaite
# from: https://stackoverflow.com/a/64448286
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from .enums.policy_type import PolicyType
    
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.0002
L2_WEIGHT = 0.0001

ACTOR_LSTM_UNITS = 256

# Strategy to utilise multiple GPUs.
#
# HierarchicalCopyAllReduce for multi-GPU setup on single machine recommendation from:
# https://github.com/y33-j3T/Coursera-Deep-Learning/blob/master/Custom%20and%20Distributed%20Training%20with%20Tensorflow/Week%204%20-%20Distributed%20Training/C2_W4_Lab_2_multi-GPU-mirrored-strategy.ipynb
#strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

device_count = strategy.num_replicas_in_sync
print('Number of devices: {}'.format(device_count))

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
        with strategy.scope():
            self.actor_model = self.get_actor()
            self.actor_model_old = self.get_actor()
            self.critic_model = self.get_critic()

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

    def get_embeddings_from_probabilities(self, probabilities, chosen_indices, use_chosen_indices):
        chosen_indices = tf.cond(
            tf.equal(use_chosen_indices, True),
            true_fn=lambda: chosen_indices,
        
            # Log probabilities as tf.random.categorical expects log probabilities.
            false_fn=lambda: tf.random.categorical(tf.math.log(probabilities), num_samples=1, dtype=tf.int32),
        ) 
        chosen_indices = tf.squeeze(chosen_indices)
        
        embeddings = tf.convert_to_tensor(self.embeddings, dtype=tf.float32)
        
        # TODO: Consider using tf.gather_nd for performance increase:
        # https://github.com/matterport/Mask_RCNN/issues/749#issuecomment-1497595166
        chosen_embeddings = tf.gather(embeddings, chosen_indices)

        probabilities = tf.cast(probabilities, dtype=tf.float64)
        chosen_probabilities = tf.gather(probabilities, chosen_indices, axis=1, batch_dims=1)
        
        return chosen_indices, chosen_embeddings, chosen_probabilities

    def __create_lstm_layer(self, units: int, return_tensors: bool = True, bidirectional: bool = True):
        '''
        Creates LTSM (or Bidirectional LSTM) with dropout.
        '''
        lstm = tf.keras.layers.LSTM(
            units,
            return_state=return_tensors,
            return_sequences=return_tensors,
        )
        
        return tf.keras.layers.Bidirectional(lstm) if bidirectional else lstm

    def __create_hidden_dense_layer(self, units: int):
        return tf.keras.layers.Dense(
            units,
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
            kernel_constraint=tf.keras.constraints.max_norm(3),
            bias_constraint=tf.keras.constraints.max_norm(3)
        )
    

    def get_actor(self):
        input_rl_state = tf.keras.layers.Input(shape=[self.state_size,])

        dense_rl_state = self.__create_hidden_dense_layer(128)(input_rl_state)
        dense_rl_state = self.__create_hidden_dense_layer(128)(dense_rl_state)
        dense_rl_state = self.__create_hidden_dense_layer(128)(dense_rl_state)

        # Average embedding input.
        input_embedding = tf.keras.layers.Input(shape=[1, self.embedding_size,],)

        lstm = self.__create_lstm_layer(ACTOR_LSTM_UNITS, bidirectional=False)(input_embedding)
        lstm = self.__create_lstm_layer(ACTOR_LSTM_UNITS, bidirectional=False)(lstm)
        lstm = self.__create_lstm_layer(ACTOR_LSTM_UNITS, return_tensors=False, bidirectional=False)(lstm)

        concat = tf.keras.layers.Concatenate()([dense_rl_state, lstm,])

        dense = self.__create_hidden_dense_layer(128)(concat)
        dense = self.__create_hidden_dense_layer(128)(dense)
        
        dense = tf.keras.layers.Dense(self.dictionary_length, activation='softmax')(dense)

        return tf.keras.Model([input_rl_state, input_embedding,], dense)

    def get_critic(self):
        input_rl_state = tf.keras.layers.Input(shape=[self.state_size,])

        dense = self.__create_hidden_dense_layer(128)(input_rl_state)
        dense = self.__create_hidden_dense_layer(128)(dense)
        dense = self.__create_hidden_dense_layer(64)(dense)
        dense = self.__create_hidden_dense_layer(32)(dense)
        dense = self.__create_hidden_dense_layer(16)(dense)
        dense = self.__create_hidden_dense_layer(8)(dense)
        dense = self.__create_hidden_dense_layer(4)(dense)
        dense = self.__create_hidden_dense_layer(2)(dense)
        output = tf.keras.layers.Dense(1)(dense)

        return tf.keras.Model([input_rl_state,], output)

    def concat_next_token_indicies(
        self, 
        actions,
        probabilities,
        batch_size,
        action_index,
        action_index_float,
        embeddings,
        type: int,
        training: bool,
        states,
        actions_reference,
    ):
        input = (states, embeddings,)

        normal_policy_id = tf.constant(PolicyType.NORMAL.value, dtype=tf.int32)

        one_hot_probabilities = tf.cond(
            tf.equal(type, normal_policy_id),
                true_fn=lambda: self.actor_model(input, training=training),
                false_fn=lambda: self.actor_model_old(input, training=training))
        
        chosen_indices, chosen_embeddings, chosen_probabilities = tf.cond(
            tf.equal(actions_reference.shape[0], tf.constant(0)),
            true_fn=lambda: self.get_embeddings_from_probabilities(one_hot_probabilities, tf.fill([batch_size, 1], -1), False),
            false_fn=lambda: self.get_embeddings_from_probabilities(one_hot_probabilities, actions_reference[:,action_index], True)
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
        
        return actions, probabilities, batch_size, action_index, action_index_float, embeddings, type, training, states, actions_reference

    # TODO/NOTE: Since batch size can be variable for actor, the shape returned from actor
    # output is unknown. If this method is decorated with @tf.function, loose shape invariants
    # must be defined in the TF while loop for field values corresponding to actor output
    # values.
    #
    # The @tf.function is taken off this method as a workaround.
    def policy(self, states, type: int, batch_size, training: bool, actions_reference: tf.Tensor):
        '''
        `type` is expected to be the enumerated value of a `PolicyType`.
        
        Setting `action_reference` to `tf.constant([])` marks as no action reference.
        This ensures tokens are stochastically chosen instead of following the reference.
        '''
        action_size = tf.constant(self.action_size, dtype=tf.int32)
        dictionary_length = tf.constant(self.dictionary_length - 1, dtype=tf.int32)

        actions = tf.fill([batch_size, action_size], dictionary_length)
        probabilities = tf.zeros([batch_size, action_size], dtype=tf.float64)
        
        embeddings = tf.zeros([batch_size, 1, self.embedding_size], dtype=tf.float32)

        action_index = tf.constant(0, dtype=tf.int32)
        action_index_float = tf.constant(0.0, dtype=tf.float32)

        actions, probabilities, *_ = tf.while_loop(
            cond=lambda *_: True,
            body=self.concat_next_token_indicies,
            loop_vars=(
                actions,
                probabilities,
                batch_size,
                action_index,
                action_index_float,
                embeddings,
                type,
                training,
                states,
                actions_reference
            ),
            maximum_iterations=action_size,
            shape_invariants=(
                actions.shape,
                probabilities.shape,
                tf.TensorShape(None),
                action_index.shape,
                action_index_float.shape,
                embeddings.shape,
                tf.TensorShape(None),
                tf.TensorShape(None),
                states.shape,
                actions_reference.shape
            ))

        action_likelihoods = tf.math.reduce_prod(probabilities, axis=-1, keepdims=True)

        return actions, action_likelihoods
