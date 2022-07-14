import torch
import torch.nn as nn
from torch.nn import functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=1 ,batch_first=True, bidirectional=True)
        self.query = nn.Linear(hidden_size * 2, 1, bias=False)
        self.init_weight()


    def init_weight(self):
        nn.init.xavier_uniform_(self.query.weight)

        for i in range(self.LSTM.num_layers):
            nn.init.orthogonal_(getattr(self.LSTM, f'weight_hh_l{i}'))
            nn.init.orthogonal_(getattr(self.LSTM, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.LSTM, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.LSTM, f'bias_ih_l{i}'), val=0)
            # getattr(self.LSTM, f'bias_hh_l{i}').data.chunk(4)[1].fill_(1)

            if self.LSTM.bidirectional:
                nn.init.orthogonal_(getattr(self.LSTM, f'weight_hh_l{i}_reverse'))
                nn.init.orthogonal_(getattr(self.LSTM, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.LSTM, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.LSTM, f'bias_ih_l{i}_reverse'), val=0)
                # getattr(self.LSTM, f'bias_hh_l{i}_reverse').data.chunk(4)[1].fill_(1)


    def forward(self, X, X_Len):
        X = nn.utils.rnn.pack_padded_sequence(X, X_Len, enforce_sorted=False, batch_first=True)
        X, _ = self.LSTM(X)
        X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = self.word_attention(X)
        return X


    def word_attention(self, X):
        A = self.query(X)
        A = torch.softmax(A, 1)
        X = torch.sum(X * A, 1)
        return X