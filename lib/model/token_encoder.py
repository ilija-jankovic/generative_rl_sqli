# Modified and pieced together from TensorFlow word2vec tutorial:
# https://www.tensorflow.org/text/tutorials/word2vec

from typing import List
import tensorflow as tf
from keras import layers
import keras
import tqdm

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences: List[List[int]], window_size: int, num_ns: int, vocab_size: int, seed: int):
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
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      context = tf.squeeze(context)
      
      label = tf.constant([1] + [0]*num_ns, dtype="int64")
      label = tf.squeeze(label)

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels


class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size: int, embedding_dim: int, num_ns: int):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots
  

def learn_embeddings(training_data: List[List[int]], vocabulary_length: int):
    num_ns = 4

    print('Generating training data...')
    targets, contexts, labels = generate_training_data(
        sequences=training_data,
        window_size=2,
        num_ns=num_ns,
        vocab_size=vocabulary_length,
        seed=11)
    
    embedding_dim = 128
    word2vec = Word2Vec(vocabulary_length, embedding_dim, num_ns=num_ns)
    word2vec.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    
    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    print('Training...')
    word2vec.fit(dataset, epochs=20)

    # TODO: Save the weights if training data is not based on table/column names.
    # Better solution is to add table/column names to training data.
    
    return word2vec.get_layer('w2v_embedding').get_weights()[0]

