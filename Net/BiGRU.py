import torch
import torch.nn as nn
from torch.nn import functional as F

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRU, self).__init__()

        self.GRU = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.query = nn.Linear(hidden_size * 2, 1, bias=False)
        self.init_weight()


    def init_weight(self):
        nn.init.xavier_uniform_(self.query.weight)
        for p in self.GRU.named_parameters():
            if 'weight' in p[0]:
                nn.init.orthogonal_(p[1])
            elif 'bias' in p[0]:
                nn.init.ones_(p[1])


    def forward(self, X, X_Len):
        X = nn.utils.rnn.pack_padded_sequence(X, X_Len, enforce_sorted=False, batch_first=True)
        X, _ = self.GRU(X)
        X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = self.word_attention(X)
        return X


    def word_attention(self, X):
        A = self.query(X)
        A = torch.softmax(A, 1)
        X = torch.sum(X * A, 1)
        return X