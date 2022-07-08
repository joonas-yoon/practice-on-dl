import torch
import torch.nn as nn

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
