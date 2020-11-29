import io
import itertools
import numpy as np
import pandas as pd
import os
import re
import string
from tqdm import tqdm

import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from utils import data_clean

# 查看tensorflow版本 ==> 2.2.0
print(tf.__version__)

def build_dataset():
    file_name = ['附件2.xlsx', '附件3.xlsx', '附件4.xlsx']
    file_path = './data/'

    data1_df = pd.read_excel(os.path.join(file_path, file_name[0]))
    data2_df = pd.read_excel(os.path.join(file_path, file_name[1]))
    data3_df = pd.read_excel(os.path.join(file_path, file_name[2]))

    data1_df = data_clean(data1_df)
    data2_df = data_clean(data2_df)
    data3_df = data_clean(data3_df)

    datasets = []
    datasets.extend(data1_df['留言详情'].to_list())
    datasets.extend(data1_df['留言主题'].to_list())

    datasets.extend(data2_df['留言详情'].to_list())
    datasets.extend(data2_df['留言主题'].to_list())

    datasets.extend(data3_df['留言详情'].to_list())
    datasets.extend(data3_df['留言主题'].to_list())

    return datasets

datasets = build_dataset()
print(datasets[0])
# 统计语料中的句子长度：
word_set = []
for words in tqdm(datasets):
    for word in words:
        if word not in word_set and len(word)>1:
            word_set.append(word)

vocab_len = len(word_set)
print(vocab_len)

# 将词转换为id
def word2id_inver(word_set):
    word2id = {}
    id2word = {}
    for index, word in enumerate(word_set):
        word2id[word] = index
        id2word[index] = word
    return word2id, id2word

word2id, id2word = word2id_inver(word_set)

def build_sequences(courpus, word2id):
    sequences = []
    for texts in tqdm(courpus):
        sent_id = []
        for word in texts:
            if len(word) > 1:
                sent_id.append(word2id[word])
            else:
                continue
        sequences.append(sent_id)
    return sequences

sequences = build_sequences(datasets, word2id)



#   

vocab_size = 126222
sequence_length = 138
SEED = 42

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence, 
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

      # Iterate over each positive skip-gram pair to produce training examples 
      # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                                     tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                                                                     true_classes=context_class,
                                                                     num_true=1, 
                                                                     num_sampled=num_ns, 
                                                                     unique=True, 
                                                                     range_max=vocab_size, 
                                                                     seed=SEED, 
                                                                     name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
            negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels

targets, contexts, labels = generate_training_data(
    sequences=sequences, 
    window_size=4, 
    num_ns=4, 
    vocab_size=vocab_size,
    seed=SEED)

print(len(targets), len(contexts), len(labels))

BATCH_SIZE = 16
BUFFER_SIZE = 10000

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)

num_ns = 4
class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(vocab_size, 
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding", )
        self.context_embedding = Embedding(vocab_size, 
                                       embedding_dim, 
                                       input_length=num_ns+1)
        self.dots = Dot(axes=(3,2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)

def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

embedding_dim = 64
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./train_logs")

word2vec.fit(dataset, epochs=30, batch_size=32)

# 获取词向量
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]

# 保存训练的词向量
out_v = io.open('./data/vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('./data/metadata.tsv', 'w', encoding='utf-8')


for index, word in id2word.items():
    if  index == 0: continue # skip 0, it's padding.
    vec = weights[index] 
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()