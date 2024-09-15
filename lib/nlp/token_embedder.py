# Modified and pieced together from TensorFlow word2vec tutorial:
# https://www.tensorflow.org/text/tutorials/word2vec

from typing import List
from .word2vec import Word2Vec
import tensorflow as tf
import tqdm
  
class TokenEmbedder:
  embedding_dim: int
  window_size: int
  num_ns: int

  def __init__(
    self,
    embedding_dim: int,
    window_size: int = 2,
    num_ns: int = 4,
  ):
    self.embedding_dim = embedding_dim
    self.window_size = window_size
    self.num_ns = num_ns

  # Generates skip-gram pairs with negative sampling for a list of sequences
  # (int-encoded sentences) based on window size, number of negative samples
  # and vocabulary size.
  def __generate_training_data(
    self,
    sequences: List[List[int]],
    vocab_size: int,
    seed: int = 11,
  ):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):

      # Generate positive skip-gram pairs for a sequence (sentence).
      positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=self.window_size,
            negative_samples=0)

      # Iterate over each positive skip-gram pair to produce training examples
      # with a positive context word and negative samples.
      for target_word, context_word in positive_skip_grams:
        context_class = tf.expand_dims(
            tf.constant([context_word], dtype="int64"), 1)
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=context_class,
            num_true=1,
            num_sampled=self.num_ns,
            unique=True,
            range_max=vocab_size,
            seed=seed,
            name="negative_sampling")

        # Build context and label vectors (for one target word)
        negative_sampling_candidates = tf.expand_dims(
            negative_sampling_candidates, 1)

        context = tf.concat([context_class, negative_sampling_candidates], 0)
        context = tf.squeeze(context)
        
        label = tf.constant([1] + [0]*self.num_ns, dtype="int64")
        label = tf.squeeze(label)

        # Append each element from the training example to global lists.
        targets.append(target_word)
        contexts.append(context)
        labels.append(label)

    return targets, contexts, labels

  def learn_embeddings(
    self,
    training_data: List[List[int]],
    vocabulary_length: int,
    batch_size: int,
    buffer_size: int,
  ) -> tf.Tensor:
      print('Generating embedding training data...')
      targets, contexts, labels = self.__generate_training_data(
          sequences=training_data,
          vocab_size=vocabulary_length,
          seed=11)
      
      word2vec = Word2Vec(vocabulary_length, self.embedding_dim, num_ns=self.num_ns)
      word2vec.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

      print('Creating training dataset...')
      dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
      dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

      print('Learning embeddings...')
      word2vec.fit(dataset, epochs=5)

      # TODO: Save the weights if training data is not based on table/column names.
      # Better solution is to add table/column names to training data.

      embeddings = word2vec.get_layer('w2v_embedding').get_weights()[0].tolist()

      # Normalize for better application of noise when learning.
      embeddings = tf.convert_to_tensor(embeddings)
      return tf.linalg.normalize(embeddings, axis=1)[0]


