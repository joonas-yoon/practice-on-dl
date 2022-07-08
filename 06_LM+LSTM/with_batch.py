import re
import numpy as np

import torch
import torch.nn as nn
from torch import norm, optim
from torch.utils.data import Dataset as BaseDataset, DataLoader
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

import warnings
warnings.filterwarnings(action='ignore')

# nltk.download('punkt')

class Model(nn.Module):
  def __init__(
    self,
    vocab_size,
    embedding_dim=128,
    lstm_size=128,
    lstm_layers=1
  ):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.lstm_inout_size = lstm_size
    self.lstm_hidden_size = lstm_size
    self.lstm_layers = lstm_layers
    self.embedding_block = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=embedding_dim
    )
    self.lstm = nn.LSTM(
      input_size=lstm_size,
      hidden_size=lstm_size,
      num_layers=lstm_layers,
      # dropout=0.2
    )
    self.fc = nn.Linear(lstm_size, vocab_size)

  def forward(self, x, prev_state):
    # print('x', x.shape)
    # print('prev_state', prev_state[0].shape, prev_state[1].shape)
    embedded = self.embedding_block(x)
    # print('embedded', embedded.shape)
    output, state = self.lstm(embedded, prev_state)
    # print('output', output.shape)
    # print('state', state[0].shape, state[1].shape)
    probs = self.fc(output)
    return probs, state

  def initial_state(self, sequence_length):
        return (torch.zeros(self.lstm_layers, sequence_length, self.lstm_inout_size),
                torch.zeros(self.lstm_layers, sequence_length, self.lstm_inout_size))


source_text = preprocessing.read_file('./aesop.txt')
preprocessing.write_file('source.txt', source_text)

# tokenizer = TreebankWordTokenizer()
# tokenized_words = tokenizer.tokenize(t)

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
  return torch.tensor(np.array([get_word_index(w) for w in text_sequence]))


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


class Dataset(BaseDataset):
  def __init__(self, text, sequence_length):
    self.text = text
    self.sentences = text.split()
    self.sequence_length = sequence_length

  def __len__(self):
    return len(self.sentences) - self.sequence_length

  def __getitem__(self, idx):
    chunk = self.sentences[idx : idx+self.sequence_length]
    chunk_next = self.sentences[idx+1 : idx+self.sequence_length+1]
    return (get_tokens(chunk), get_tokens(chunk_next))


sequence_length = 20
batch_size = 8

dataset = Dataset(source_text, sequence_length=sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size)

model = Model(vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

state_h, state_c = model.initial_state(sequence_length)

result = ''

# print(dict(freq))

pbar = tqdm(total=len(dataloader))
losses = []
samples = []
preview = '...'

model.train()
for step, (x, y) in enumerate(dataloader):
  try:
    optimizer.zero_grad()

    probs, (state_h, state_c) = model(x, (state_h, state_c))

    probs_true_next = torch.tensor(np.array([get_probs_of_words(y_i) for y_i in y]))
    probs_true_next = torch.tensor(probs_true_next).reshape(batch_size, sequence_length, -1)

    # print('probs', probs.shape)
    # print('probs_true_next', probs_true_next.shape)

    loss = criterion(probs, probs_true_next)

    state_h = state_h.detach()
    state_c = state_c.detach()

    loss.backward()
    optimizer.step()

    next_word_prob = probs[0][0] # first word of first sample
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


preprocessing.write_file('output.txt', result)

sns.lineplot(x=range(len(losses)), y=losses)
plt.show()
