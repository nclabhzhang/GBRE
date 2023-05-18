import torch
import torch.nn as nn
from torch.nn import functional as F
from Net import CNN, BiGRU, BiLSTM, PCNN, BioBERT
import numpy as np

class Model(nn.Module):
    def __init__(self, pre_word_vec, rel_num, opt):
        super(Model, self).__init__()

        word_embedding = torch.from_numpy(np.load(pre_word_vec))

        hidden_size = opt['hidden_size']
        pos_len = opt['max_pos_length']
        pos_dim = opt['pos_dim']             # 10
        word_dim = word_embedding.shape[1]   # 200
        que_dim = word_embedding.shape[1]    # 200
        emb_dim = word_dim * 3 + pos_dim * 2
        dataset = opt['dataset']

        self.max_sentence_len = opt['max_sentence_length']
        self.encoder_name = opt['encoder']

        if self.encoder_name.lower() == 'biobert':
            out_dim = 768
            self.att_weight_s = nn.Linear(out_dim, 1)
            self.att_weight_q = nn.Linear(out_dim, 1)
            self.att_weight_sq = nn.Linear(out_dim, 1)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze=False, padding_idx=-1)
            self.pos1_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)
            self.pos2_embedding = nn.Embedding(2 * pos_len + 1, pos_dim)

            self.att_weight_s = nn.Linear(word_dim, 1)
            self.att_weight_q = nn.Linear(word_dim, 1)
            self.att_weight_sq = nn.Linear(word_dim, 1)

        self.drop = nn.Dropout(opt['dropout'])
        self.drop_att = nn.Dropout(opt['dropout_att'])

        if self.encoder_name == 'CNN':
            self.encoder = CNN(emb_dim, hidden_size)
            self.rel = nn.Linear(hidden_size, rel_num)

        elif self.encoder_name == 'PCNN':
            self.encoder = PCNN(emb_dim, hidden_size)
            self.rel = nn.Linear(hidden_size * 3, rel_num)

        elif self.encoder_name.lower() == 'biobert':
            self.encoder = BioBERT(out_dim, dataset)
            self.rel = nn.Linear(out_dim, rel_num)

        elif self.encoder_name == 'BiGRU':
            self.encoder = BiGRU(emb_dim, hidden_size)
            self.rel = nn.Linear(hidden_size * 2, rel_num)

        elif self.encoder_name == 'BiLSTM':
            self.encoder = BiLSTM(emb_dim, hidden_size)
            self.rel = nn.Linear(hidden_size * 2, rel_num)

        self.init_weight()


    def forward(self, data):
        X_Rel = None
        if self.encoder_name.lower() == 'biobert':
            if len(data) == 8:
                X, X_Pos1, X_Pos2, X_Mask, Q_Mask, X_Scope, Q, X_Rel = data
            else:
                X, X_Pos1, X_Pos2, X_Mask, Q_Mask, X_Scope, Q = data
            del data
            X = torch.cat([X, Q], -1)
            X_Mask = torch.cat([X_Mask, Q_Mask], -1)
            del Q, Q_Mask
            X = self.encoder(X, X_Pos1, X_Pos2, X_Mask)
        else:
            if len(data) == 8:
                X, X_Pos1, X_Pos2, X_Mask, X_Len, X_Scope, Q, X_Rel = data
            else:
                X, X_Pos1, X_Pos2, X_Mask, X_Len, X_Scope, Q = data
            X = self.word_embedding(X)  # (batch, s_len, emb_dim)
            Q = self.word_embedding(Q)  # (batch, q_len, pos_dim)
            X_Pos1, X_Pos2 = self.pos1_embedding(X_Pos1), self.pos2_embedding(X_Pos2)  # (batch, s_len, pos_dim)
            X = self.qs_att(X, Q)
            X = torch.cat((X, X_Pos1, X_Pos2), -1)

            if self.encoder_name == 'CNN':
                X = self.encoder(X)
            elif self.encoder_name == 'PCNN':
                X = self.encoder(X, X_Mask)
            elif self.encoder_name == 'BiGRU':
                X = self.encoder(X, X_Len)
            elif self.encoder_name == 'BiLSTM':
                X = self.encoder(X, X_Len)

        X = self.bag_self_att(X, X_Scope)

        if self.encoder_name.lower() != 'biobert':
            X = self.drop(X)

        X = self.selective_attention(X, X_Scope, X_Rel)
        # X = self.sentence_ave_attention(X, X_Scope, X_Rel)

        return X


    def init_weight(self):
        if self.encoder_name.lower() != 'biobert':
            nn.init.xavier_uniform_(self.pos1_embedding.weight)
            nn.init.xavier_uniform_(self.pos2_embedding.weight)

        nn.init.xavier_uniform_(self.att_weight_s.weight)
        nn.init.zeros_(self.att_weight_s.bias)
        nn.init.xavier_uniform_(self.att_weight_q.weight)
        nn.init.zeros_(self.att_weight_q.bias)
        nn.init.xavier_uniform_(self.att_weight_sq.weight)
        nn.init.zeros_(self.att_weight_sq.bias)

        nn.init.xavier_uniform_(self.rel.weight)
        nn.init.zeros_(self.rel.bias)


    def qs_att(self, s, q):   # c is sentence context, q is question
        s_len = s.size(1)     # (batch, s_len, emd_dim)
        q_len = q.size(1)     # (batch, q_len, emd_dim)
        sq = []
        for i in range(q_len):
            qi = q.select(1, i).unsqueeze(1)            # (batch, 1, emb_dim)
            si = self.att_weight_sq(s * qi).squeeze()   # (batch, s_len, 1)
            sq.append(si)
        sq = torch.stack(sq, dim=-1)  # (batch, s_len, q_len)

        h = self.att_weight_s(s).expand(-1, -1, q_len) + self.att_weight_q(q).permute(0, 2, 1).expand(-1, s_len, -1) + sq   # (batch, s_len, q_len)

        a = F.softmax(h, dim=2)     # (batch, s_len, q_len)
        s2q_att = torch.bmm(a, q)   # (batch, s_len, q_len) * (batch, q_len, emb_dim) -> (batch, s_len, emb_dim)
        b = F.softmax(torch.max(h, dim=2)[0], dim=1).unsqueeze(1)   # (batch, 1, s_len)
        q2s_att = torch.bmm(b, s).squeeze()      # (batch, 1, s_len) * (batch, s_len, emb_dim) -> (batch, emb_dim)
        del q
        q2s_att = q2s_att.unsqueeze(1).expand(-1, s_len, -1)    # (batch, s_len, emb_dim) (tiled)

        X = torch.cat([s, s * s2q_att, s * q2s_att], dim=-1)     # (batch, s_len, emb_dim * 3)

        return X


    def bag_self_att(self, X, scope):
        bag_num = len(scope)
        dim = X.shape[-1]
        output = []
        for i in range(bag_num):
            cur_bag_sentence = X[scope[i][0]:scope[i][1]].view(-1, dim)
            tmp_res = torch.matmul(cur_bag_sentence, cur_bag_sentence.transpose(-1, -2)) / np.sqrt(dim)
            attention_weights = nn.Softmax(dim=-1)(tmp_res)
            attention_weights = self.drop_att(attention_weights)
            tmp_res = torch.matmul(attention_weights, cur_bag_sentence)
            output.append(tmp_res)
        output = torch.cat(output, 0)
        return output


    def selective_attention(self, X, X_Scope, Rel=None):
        bag_output = []
        if Rel is not None:  # For training
            Rel = F.embedding(Rel, self.rel.weight)
            for i in range(X_Scope.shape[0]):
                bag_rep = X[X_Scope[i][0]: X_Scope[i][1]]
                att_score = F.softmax(bag_rep.matmul(Rel[i]), 0).view(1, -1)  # (1, Bag_size)
                att_output = att_score.matmul(bag_rep)  # (1, dim)
                bag_output.append(att_output.squeeze())  # (dim, )
            bag_output = torch.stack(bag_output)
            bag_output = self.drop(bag_output)
            bag_output = self.rel(bag_output)
        else:  # For testing
            att_score = X.matmul(self.rel.weight.t())  # (Batch_size, dim) -> (Batch_size, R)
            for s in X_Scope:
                bag_rep = X[s[0]:s[1]]  # (Bag_size, dim)
                bag_score = F.softmax(att_score[s[0]:s[1]], 0).t()  # (R, Bag_size)
                att_output = bag_score.matmul(bag_rep)  # (R, dim)
                bag_output.append(torch.diagonal(F.softmax(self.rel(att_output), -1)))
            bag_output = torch.stack(bag_output)
            # bag_output = F.softmax(bag_output, -1)
        return bag_output


    def sentence_ave_attention(self, X, X_Scope, Rel=None):
        bag_output = []
        if Rel is not None:  # For training
            for i in range(X_Scope.shape[0]):
                bag_rep = X[X_Scope[i][0]: X_Scope[i][1]]  # (Bag_size, dim)
                bag_output.append(bag_rep.mean(0))  # (dim, )
            bag_output = torch.stack(bag_output)
            bag_output = self.drop(bag_output)
            bag_output = self.rel(bag_output)
        else:  # For testing
            for s in X_Scope:
                bag_rep = X[s[0]:s[1]]  # (Bag_size, dim)
                ave_output = bag_rep.mean(0)  # (dim, )
                bag_output.append(F.softmax(self.rel(ave_output), -1))
            bag_output = torch.stack(bag_output)
        return bag_output