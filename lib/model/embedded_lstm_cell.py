import tensorflow as tf
from typing import List
from keras.layers.rnn import lstm
from keras import backend as K

class EmbeddedLSTMCell(lstm.LSTMCell):

    embeddings: List[List[float]]
    embedding_size: int
    sess = K.get_session()

    def __init__(self, units, embeddings: List[List[float]], embedding_size: int, **kwargs):
        super().__init__(units, **kwargs)
        
        self.embeddings = embeddings
        self.embedding_size = embedding_size

    @tf.function
    def __set_embedding(self, h_tensor: tf.TensorArray):
        h_tensor = tf.unstack(h_tensor)

        last_h = h_tensor[-1]
        
        if last_h < 0.0 or last_h >= len(self.embeddings):
            last_h = tf.convert_to_tensor([0.0] * self.embedding_size)
        else:
            last_h = tf.gather(self.embeddings, [tf.cast(last_h, tf.int32)])

        last_h = tf.squeeze(last_h)

        return tf.concat([h_tensor[:-1], last_h], axis=0)

    def call(self, inputs, states, training=None):
        h, outputs = super().call(inputs, states, training)

        outputs[0] = tf.map_fn(lambda h_tensor: self.__set_embedding(h_tensor), h)

        return h, outputs

