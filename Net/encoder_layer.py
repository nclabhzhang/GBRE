import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, BertModel


class CNN(nn.Module):
    def __init__(self, emb_dim, hidden_size):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(emb_dim, hidden_size, kernel_size=3, padding=1)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, X):
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X, _ = torch.max(X, 1)
        X = F.relu(X)
        return X


class PCNN(nn.Module):
    def __init__(self, emb_dim, hidden_size):
        super(PCNN, self).__init__()
        mask_embedding = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding)
        self.cnn = nn.Conv1d(emb_dim, hidden_size, kernel_size=3, padding=1)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)

    def forward(self, X, X_mask):
        X = self.cnn(X.transpose(1, 2)).transpose(1, 2)
        X = self.pool(X, X_mask)
        X = F.relu(X)
        return X

    def pool(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        hidden_size = X.shape[-1]
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, hidden_size * 3)


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


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.query = nn.Linear(hidden_size * 2, 1, bias=False)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.query.weight)
        for i in range(self.LSTM.num_layers):
            nn.init.orthogonal_(getattr(self.LSTM, f'weight_hh_l{i}'))
            nn.init.orthogonal_(getattr(self.LSTM, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.LSTM, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.LSTM, f'bias_ih_l{i}'), val=0)
            if self.LSTM.bidirectional:
                nn.init.orthogonal_(getattr(self.LSTM, f'weight_hh_l{i}_reverse'))
                nn.init.orthogonal_(getattr(self.LSTM, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.LSTM, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.LSTM, f'bias_ih_l{i}_reverse'), val=0)

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


class BioBERT(nn.Module):
    def __init__(self, hidden_size=768, dataset='biorel'):
        super(BioBERT, self).__init__()
        self.dataset = dataset.lower()
        if self.dataset == 'biorel':
            self.biobert = BertModel.from_pretrained('bert-base-uncased')
            self.linear = nn.Linear(hidden_size * 2, hidden_size)
            self.init_weight()
        else:
            self.biobert = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, X, X_Pos1, X_Pos2, X_mask):
        if self.dataset == 'biorel':
            X = self.biobert(X, attention_mask=X_mask)
            X = X[0]
            # Get entity start hidden state
            onehot_head = torch.zeros(X.shape[:2], dtype=torch.float32, device=X.device)
            onehot_tail = torch.zeros(X.shape[:2], dtype=torch.float32, device=X.device)
            onehot_head = onehot_head.scatter_(1, X_Pos1.unsqueeze(1), 1)
            onehot_tail = onehot_tail.scatter_(1, X_Pos2.unsqueeze(1), 1)
            head_hidden = (onehot_head.unsqueeze(2) * X).sum(1)
            tail_hidden = (onehot_tail.unsqueeze(2) * X).sum(1)
            X = torch.cat([head_hidden, tail_hidden], 1)
            X = self.linear(X)
        else:
            X = self.biobert(X, attention_mask=X_mask)
            X = X[1]
        return X