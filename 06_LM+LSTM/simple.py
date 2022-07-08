import re
import numpy as np

import torch
import torch.nn as nn
from torch import norm, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torchtext.vocab import vocab
from collections import Counter

import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.lm import Vocabulary

from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import preprocessing
from preprocessing import TOKENS, UNKNOWN_TOKEN

from model import Model

# nltk.download('punkt')

source_text = preprocessing.read_file('./aesop.txt')
preprocessing.write_file('source.txt', source_text)

# tokenizer = TreebankWordTokenizer()
# tokenized_words = tokenizer.tokenize(t)

tokenized_words = word_tokenize(source_text)
tokenized_words = word_tokenize(source_text)
freq = FreqDist(tokenized_words)
freq.update(TOKENS)
word_to_index = {}
index_to_word = {}
for word in freq:
  idx = len(word_to_index)
  word_to_index[word] = idx
  index_to_word[idx] = word
vocab_size = len(word_to_index)


def get_word_index(word):
  unk = word_to_index[UNKNOWN_TOKEN]
  return word_to_index[word] if word in word_to_index else unk

def get_tokens(text_sequence: list):
  return np.array([get_word_index(w) for w in text_sequence])


def get_probs_of_words(text_sequence: list):
  total = sum(freq.values())
  n_seq = len(text_sequence)
  z = np.zeros((n_seq, vocab_size))
  for i, word in enumerate(text_sequence):
    z[i][get_word_index(word)] = freq[word] / total
  # normalize
  def normalize(v):
    if np.sum(v) == 0:
      zero = np.zeros((vocab_size))
      zero[get_word_index(UNKNOWN_TOKEN)] = 1.0
      return zero
    return v / np.sqrt(np.sum(v**2))
  return np.apply_along_axis(normalize, 1, z)


sentences = source_text.split()
sequence_length = 10
print('total words:', len(sentences))
print('vocab_size:', vocab_size)

model = Model(vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
state_h, state_c = model.initial_state(sequence_length)

result = ''

# print(dict(freq))

seq_iters = len(sentences) - sequence_length

pbar = tqdm(total=seq_iters)
losses = []
samples = []
preview = '...'

for step, i in enumerate(range(seq_iters)):
  try:
    optimizer.zero_grad()

    chunk = sentences[i:i+sequence_length]
    chunk_next = sentences[i+1:i+sequence_length+1]

    seq = get_tokens(chunk)
    seq = torch.tensor(seq).reshape(1, -1)
    y_true = get_tokens(chunk_next)
    y_true = torch.tensor(y_true).reshape(1, -1)
    probs, (state_h, state_c) = model(seq, (state_h, state_c))

    probs_true_next = get_probs_of_words(chunk_next)
    probs_true_next = torch.tensor(probs_true_next).reshape(1, sequence_length, -1)

    # print('chunk', seq.shape)
    # print('probs', probs.shape)
    # print('y_true', y_true.shape)
    # print('probs_true_next', probs_true_next.shape)

    loss = criterion(probs, probs_true_next)

    state_h = state_h.detach()
    state_c = state_c.detach()

    loss.backward()
    optimizer.step()

    next_word_prob = probs.reshape(sequence_length, -1)[0]
    next_word_index = next_word_prob.argmax().detach().item()
    next_word = index_to_word[next_word_index]
    result += next_word + ' '
    samples.append(next_word)

    if step % 50 == 0:
      preview = ' '.join(samples[-5:])
      del samples
      samples = []

    losses.append(loss.item())
    pbar.set_description('\"... {}\" | loss: {:.8f}'.format(preview, loss.item()))
    pbar.update()
  except:
    break


preprocessing.write_file('output_simple.txt', result)

sns.lineplot(x=range(len(losses)), y=losses)
plt.show()
