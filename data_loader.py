import os
import json
import torch
import pickle
import numpy as np
import torch.utils.data as data
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
os.environ['TOKOENIZERS_PARALLELISM'] = 'true'


class Bag:
    def __init__(self):
        self.word = []
        self.pos1 = []
        self.pos2 = []
        self.mask = []
        self.mask_q = []
        self.length = []
        self.rel = []
        self.question = []


class Dataset(data.Dataset):
    def _preprocess(self):
        # Load files
        print("Loading data file...")
        ori_data = json.load(open(self.data_dir, "r"))
        print("Finish loading")
        # Sort data by entities and relations
        print("Sort data...")
        ori_data.sort(key=lambda a: a['head']['CUI'] + '#' + a['tail']['CUI'] + '#' + str(a['relation']))
        print("Finish sorting")
        # Pre-process data
        print("Pre-processing data...")
        bag_sentence_num = 0
        last_bag = None
        self.data = []
        bag = Bag()
        processor = self.biobert_processor if self.tokenizer else self.vanilla_processor

        for ins in tqdm(ori_data):
            _rel = self.rel2id[ins['relation']] if ins['relation'] in self.rel2id else self.rel2id['NA']
            if self.training:
                cur_bag = (ins['head']['CUI'], ins['tail']['CUI'], ins['relation'])  # used for train
            else:
                cur_bag = (ins['head']['CUI'], ins['tail']['CUI'])  # used for test

            if cur_bag != last_bag:
                if last_bag is not None:
                    self.data.append(bag)
                    bag = Bag()
                last_bag = cur_bag

            bag.rel.append(_rel)
            processor(bag, ins)

        # append the last bag
        if last_bag is not None:
            self.data.append(bag)

        print("Finish pre-processing")
        print("Storing processed files...")
        pickle.dump(self.data, open(os.path.join(self.processed_data_dir, self.processed_data_name), 'wb'))
        print("Finish storing")


    def __init__(self, file_name, opt, training=True):
        super().__init__()
        self.dataset = os.path.join(opt['root'], opt['dataset'])
        self.data_name = opt['dataset']
        self.file_name = file_name
        self.processed_data_dir = os.path.join(opt['processed_data_dir'], opt['dataset'])
        self.processed_data_name = opt['encoder'] + '-' + self.file_name.split(".")[0] + '.pkl'
        self.data_dir = os.path.join(self.dataset, self.file_name)
        self.rel_dir = os.path.join(self.dataset, opt['rel'])
        self.vec_dir = os.path.join(self.dataset, opt['vec'])
        self.max_pos_length = opt['max_pos_length']
        self.max_sentence_length = opt['max_sentence_length']
        self.max_question_length = opt['max_question_length']
        self.bag_size = opt['bag_size']
        self.training = training
        self.vec_save_dir = os.path.join(self.processed_data_dir, 'word_vec.npy')
        self.word2id_save_dir = os.path.join(self.processed_data_dir, 'word2id.json')
        if opt['encoder'].lower() == 'biobert':
            if self.data_name.lower() == 'biorel':
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            else:
                self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        else:
            self.tokenizer = None
        self.init_rel()

        if not os.path.exists(opt['processed_data_dir']):
            os.mkdir(os.path.join(opt['processed_data_dir']))
        if not os.path.exists(self.processed_data_dir):
            os.mkdir(os.path.join(self.processed_data_dir))

        if os.path.exists(self.vec_save_dir) and os.path.exists(self.word2id_save_dir):
            self.word2id = json.load(open(self.word2id_save_dir))
        else:
            print("Extracting word2vec data")
            self.init_word()

        try:
            print("Trying to load processed data")
            self.data = pickle.load(open(os.path.join(self.processed_data_dir, self.processed_data_name), 'rb'))
            print("Load successfully")
        except:
            print("Processed data does not exist")
            self._preprocess()
        print("bag num:", self.__len__())


    def init_rel(self):
        self.rel2id = json.load(open(self.rel_dir, "r"))


    def rel_num(self):
        return len(self.rel2id)


    def init_word(self):
        ori_word_vec = json.load(open(self.vec_dir, "r"))
        self.word2id = {}
        num = len(ori_word_vec)
        dim = len(ori_word_vec[0]['vec'])
        UNK = num
        PAD = num + 1
        word_vec = np.zeros((num+2, dim), dtype=np.float32)
        for cur_id, word in enumerate(ori_word_vec):
            w = word['word'].lower()
            self.word2id[w] = cur_id
            word_vec[cur_id, :] = word['vec']
        word_vec[num] = np.random.randn(dim) / np.sqrt(dim)
        word_vec[num+1] = np.zeros(dim)
        self.word2id['[UNK]'] = UNK
        self.word2id['[PAD]'] = PAD
        np.save(self.vec_save_dir, word_vec)
        json.dump(self.word2id, open(self.word2id_save_dir, 'w'))


    def last_bag_processor(self, bag):
        bag.ent1 = list(set(bag.ent1))
        bag.ent2 = list(set(bag.ent2))
        if self.word2id['[UNK]'] in bag.ent1 and len(bag.ent1) > 1:
            bag.ent1.remove(self.word2id['[UNK]'])
        if self.word2id['[UNK]'] in bag.ent2 and len(bag.ent2) > 1:
            bag.ent2.remove(self.word2id['[UNK]'])


    def vanilla_processor(self, bag, ins):
        # ent
        ent1 = ins['head']['word']
        ent2 = ins['tail']['word']
        _ent1 = self.word2id[ent1] if ent1 in self.word2id else self.word2id['[UNK]']
        _ent2 = self.word2id[ent2] if ent2 in self.word2id else self.word2id['[UNK]']
        bag.ent1.append(_ent1)
        bag.ent2.append(_ent2)

        # word
        cur_sentence_length = len(ins['sentence'].split())
        if cur_sentence_length > self.max_sentence_length:
            idx_head, idx_tail = ins['head']['start'], ins['tail']['start'] + len(ent2)
            sentence = ins['sentence'][idx_head:idx_tail].split()
        else:
            sentence = ins['sentence'].split()
        _ids = [self.word2id[word] if word in self.word2id else self.word2id['[UNK]'] for word in sentence]
        _ids = _ids[:self.max_sentence_length]
        _ids.extend([self.word2id['[PAD]'] for _ in range(self.max_sentence_length - len(sentence))])
        bag.word.append(_ids)

        # pos
        if cur_sentence_length > self.max_sentence_length:
            p1 = sentence.index(ent1) if ent1 in sentence else 0
            p2 = sentence.index(ent2) if ent2 in sentence else 0
        else:
            p1 = ins['head']['split_start']
            p2 = ins['tail']['split_start']
        p1 = p1 if p1 < self.max_sentence_length else self.max_sentence_length - 1
        p2 = p2 if p2 < self.max_sentence_length else self.max_sentence_length - 1
        _pos1 = np.arange(self.max_sentence_length) - p1 + self.max_pos_length
        _pos2 = np.arange(self.max_sentence_length) - p2 + self.max_pos_length

        _pos1[_pos1 > 2 * self.max_pos_length] = 2 * self.max_pos_length
        _pos1[_pos1 < 0] = 0
        _pos2[_pos2 > 2 * self.max_pos_length] = 2 * self.max_pos_length
        _pos2[_pos2 < 0] = 0

        bag.pos1.append(_pos1.tolist())
        bag.pos2.append(_pos2.tolist())

        # question
        ent1 = ins['head']['word']
        ent2 = ins['tail']['word']
        auto_que = f'what is the relation between {ent1} and {ent2} ?'.split()
        que_ids = [self.word2id[word] if word in self.word2id else self.word2id['[UNK]'] for word in auto_que]
        que_len = len(que_ids)
        for i in range(self.max_question_length - que_len):
            que_ids.append(0)  # Padding
        que_ids = que_ids[:self.max_question_length]
        bag.question.append(que_ids)

        # mask
        p1, p2 = sorted((p1, p2))
        _mask = np.zeros(self.max_sentence_length, dtype=np.long)
        _mask[p2 + 1: len(sentence)] = 3
        _mask[p1 + 1: p2 + 1] = 2
        _mask[:p1 + 1] = 1
        _mask[len(sentence):] = 0
        bag.mask.append(_mask)

        # sentence length
        _length = min(len(sentence), self.max_sentence_length)
        bag.length.append(_length)


    def biobert_processor(self, bag, ins):
        ent1 = ins['head']['word']
        ent2 = ins['tail']['word']
        cur_sentence_length = len(ins['sentence'].split())
        if cur_sentence_length > self.max_sentence_length:
            idx_head, idx_tail = ins['head']['start'], ins['tail']['start'] + len(ent2)
            sentence = ins['sentence'][idx_head:idx_tail].split()
        else:
            sentence = ins['sentence'].split()
        if cur_sentence_length > self.max_sentence_length:
            p1 = sentence.index(ent1) if ent1 in sentence else 0
            p2 = sentence.index(ent2) if ent2 in sentence else 0
        else:
            p1 = ins['head']['split_start']
            p2 = ins['tail']['split_start']
        p1 = p1 if p1 < self.max_sentence_length else self.max_sentence_length - 1
        p2 = p2 if p2 < self.max_sentence_length else self.max_sentence_length - 1
        rev = False
        if p1 > p2:
            p1, p2 = p2, p1
            rev = True
        sent1 = self.tokenizer.tokenize(' '.join(sentence[:p1]))
        ent1 = self.tokenizer.tokenize(ent1)
        sent2 = self.tokenizer.tokenize(' '.join(sentence[p1 + 1:p2]))
        ent2 = self.tokenizer.tokenize(ent2)
        sent3 = self.tokenizer.tokenize(' '.join(sentence[p2 + 1:]))
        if self.data_name.lower() == 'biorel':
            if not rev:
                tokens = ['[CLS]'] + sent1 + ['[unused0]'] + ent1 + ['[unused1]'] + sent2 + ['[unused2]'] + ent2 + ['[unused3]'] + sent3 + ['[SEP]']
            else:
                tokens = ['[CLS]'] + sent1 + ['[unused2]'] + ent1 + ['[unused3]'] + sent2 + ['[unused0]'] + ent2 + ['[unused1]'] + sent3 + ['[SEP]']
            _ids = self.tokenizer.convert_tokens_to_ids(tokens)
            p1 = min(1 + len(sent1), self.max_sentence_length - 1)
            p2 = min(3 + len(sent1) + len(ent1) + len(sent2), self.max_sentence_length - 1)
        else:
            if not rev:
                tokens = ['[CLS]'] + sent1 + ent1 + sent2 + ent2 + sent3 + ['[SEP]']
            else:
                tokens = ['[CLS]'] + sent1 + ent1 + sent2 + ent2 + sent3 + ['[SEP]']
            _ids = self.tokenizer.convert_tokens_to_ids(tokens)
            p1 = min(len(sent1), self.max_sentence_length - 1)
            p2 = min(len(sent1) + len(ent1) + len(sent2), self.max_sentence_length - 1)
        if rev:
            p1, p2 = p2, p1
        bag.pos1.append(p1)
        bag.pos2.append(p2)

        # question
        ent1 = ins['head']['word']
        ent2 = ins['tail']['word']
        auto_que = f'what is the relation between {ent1} and {ent2} ?'.split()
        question = self.tokenizer.tokenize(' '.join(auto_que))
        que_tokens = question + ['[SEP]']
        que_ids = self.tokenizer.convert_tokens_to_ids(que_tokens)
        que_len = len(que_ids)
        for i in range(self.max_question_length - que_len):
            que_ids.append(0)  # Padding
        que_ids = que_ids[:self.max_question_length]
        bag.question.append(que_ids)
        # question_mask
        _mask_q = np.zeros(self.max_question_length, dtype=np.long)
        _mask_q[:que_len] = 1
        bag.mask_q.append(_mask_q.tolist())


        length = len(_ids)
        _length = min(length, self.max_sentence_length)
        bag.length.append(_length)

        for i in range(self.max_sentence_length - length):
            _ids.append(0)  # Padding
        _ids = _ids[:self.max_sentence_length]
        bag.word.append(_ids)

        _length = min(len(sentence), self.max_sentence_length)

        # mask
        _mask = np.zeros(self.max_sentence_length, dtype=np.long)  # (1, L)
        _mask[:length] = 1
        bag.mask.append(_mask.tolist())


    def loss_weight(self):
        print("Calculating the class weight")
        rel_ins = []
        for bag in self.data:
            rel_ins.extend(bag.rel)
        stat = Counter(rel_ins)
        class_weight = torch.ones(self.rel_num(), dtype=torch.float32)
        for k, v in stat.items():
            class_weight[k] = 1. / v ** 0.05
        if torch.cuda.is_available():
            class_weight = class_weight.cuda()
        return class_weight


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        bag = self.data[index]
        sen_num = len(bag.word)
        select = torch.arange(sen_num)
        if self.tokenizer and sen_num > self.bag_size:
            select = torch.tensor(np.random.choice(sen_num, self.bag_size, replace=False), dtype=torch.long)
        word = torch.tensor(bag.word, dtype=torch.long)[select]
        pos1 = torch.tensor(bag.pos1, dtype=torch.long)[select]
        pos2 = torch.tensor(bag.pos2, dtype=torch.long)[select]
        mask = torch.tensor(bag.mask, dtype=torch.long)[select]
        que = torch.tensor(bag.question, dtype=torch.long)[select]
        length = torch.tensor(bag.length, dtype=torch.long)[select]
        if self.training:
            rel = torch.tensor(bag.rel[0], dtype=torch.long)
        else:
            rel = torch.zeros(len(self.rel2id), dtype=torch.long)
            for i in set(bag.rel):
                rel[i] = 1
        if self.tokenizer:
            mask_q = torch.tensor(bag.mask_q, dtype=torch.long)[select]
            return word, pos1, pos2, mask, mask_q, que, rel, length

        return word, pos1, pos2, mask, length, que, rel



def collate_fn(X):
    X = list(zip(*X))
    if len(X) == 8:
        word, pos1, pos2, mask, mask_q, que, rel, _ = X
    else:
        word, pos1, pos2, mask, length, que, rel = X

    scope = []
    ind = 0
    for w in word:
        scope.append((ind, ind + len(w)))
        ind += len(w)
    scope = torch.tensor(scope, dtype=torch.long)

    word = torch.cat(word, 0)
    pos1 = torch.cat(pos1, 0)
    pos2 = torch.cat(pos2, 0)
    mask = torch.cat(mask, 0)
    rel = torch.stack(rel)
    que = torch.cat(que, 0)
    if len(X) == 8:
        mask_q = torch.cat(mask_q, 0)
        return word, pos1, pos2, mask, mask_q, scope, que, rel

    length = torch.cat(length, 0)


    return word, pos1, pos2, mask, length, scope, que, rel


def data_loader(data_file, opt, shuffle, training=True, num_workers=4):
    dataset = Dataset(data_file, opt, training)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt['batch_size'],
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader