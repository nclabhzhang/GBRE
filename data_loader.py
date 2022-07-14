import os
import json
import torch
import pickle
import numpy as np
import torch.utils.data as data
from collections import Counter
from tqdm import tqdm


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
        bag = {
            'word': [],
            'rel': [],
            'ent1': [],
            'ent2': [],
            'pos1': [],
            'pos2': [],
            'query': [],
            'mask': [],
            'length': [],
        }

        for ins in tqdm(ori_data):
            if self.training:
                cur_bag = (ins['head']['CUI'], ins['tail']['CUI'], ins['relation'])  # used for train
            else:
                cur_bag = (ins['head']['CUI'], ins['tail']['CUI'])  # used for test

            if cur_bag != last_bag:
                if last_bag is not None:
                    bag['ent1'] = list(set(bag['ent1']))
                    bag['ent2'] = list(set(bag['ent2']))
                    if self.word2id['[UNK]'] in bag['ent1'] and len(bag['ent1']) > 1:
                        bag['ent1'].remove(self.word2id['[UNK]'])
                    if self.word2id['[UNK]'] in bag['ent2'] and len(bag['ent2']) > 1:
                        bag['ent2'].remove(self.word2id['[UNK]'])

                    self.data.append(bag)

                    bag = {
                        'word': [],
                        'rel': [],
                        'ent1': [],
                        'ent2': [],
                        'pos1': [],
                        'pos2': [],
                        'query': [],
                        'mask': [],
                        'length': [],
                    }
                last_bag = cur_bag

            # relation
            _rel = self.rel2id[ins['relation']] if ins['relation'] in self.rel2id else self.rel2id['NA']
            bag['rel'].append(_rel)

            # ent
            ent1 = ins['head']['word']
            ent2 = ins['tail']['word']
            _ent1 = self.word2id[ent1] if ent1 in self.word2id else self.word2id['[UNK]']
            _ent2 = self.word2id[ent2] if ent2 in self.word2id else self.word2id['[UNK]']
            bag['ent1'].append(_ent1)
            bag['ent2'].append(_ent2)

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
            bag['word'].append(_ids)

            # position
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

            bag['pos1'].append(_pos1)
            bag['pos2'].append(_pos2)

            # query
            if bag['query'] == []:
                query = 'what is the relation between [PAD] and [PAD] ?'.split()
                bag['query'] = [self.word2id[word] if word in self.word2id else self.word2id['[UNK]'] for word in query]

            # mask
            p1, p2 = sorted((p1, p2))
            _mask = np.zeros(self.max_sentence_length, dtype=np.long)
            _mask[p2 + 1: len(sentence)] = 3
            _mask[p1 + 1: p2 + 1] = 2
            _mask[:p1 + 1] = 1
            _mask[len(sentence):] = 0
            bag['mask'].append(_mask)

            # sentence length
            _length = min(len(sentence), self.max_sentence_length)
            bag['length'].append(_length)


        # append the last bag
        if last_bag is not None:
            bag['ent1'] = list(set(bag['ent1']))
            bag['ent2'] = list(set(bag['ent2']))
            if self.word2id['[UNK]'] in bag['ent1'] and len(bag['ent1']) > 1:
                bag['ent1'].remove(self.word2id['[UNK]'])
            if self.word2id['[UNK]'] in bag['ent2'] and len(bag['ent2']) > 1:
                bag['ent2'].remove(self.word2id['[UNK]'])
            self.data.append(bag)

        print("Finish pre-processing")
        print("Storing processed files...")
        pickle.dump(self.data, open(os.path.join(self.processed_data_dir, self.file_name.split(".")[0]+'.pkl'), 'wb'))
        print("Finish storing")


    def __init__(self, file_name, opt, training=True):
        super().__init__()
        self.file_name = file_name
        self.processed_data_dir = opt['processed_data_dir']
        self.data_dir = os.path.join(opt['root'], self.file_name)
        self.rel_dir = os.path.join(opt['root'], opt['rel'])
        self.vec_dir = os.path.join(opt['root'], opt['vec'])
        self.max_pos_length = opt['max_pos_length']
        self.max_sentence_length = opt['max_sentence_length']
        self.training = training
        self.vec_save_dir = os.path.join(self.processed_data_dir, 'word_vec.npy')
        self.word2id_save_dir = os.path.join(self.processed_data_dir, 'word2id.json')
        self.init_rel()

        if not os.path.exists(self.processed_data_dir):
            os.mkdir(os.path.join(self.processed_data_dir))

        if os.path.exists(self.vec_save_dir) and os.path.exists(self.word2id_save_dir):
            self.word2id = json.load(open(self.word2id_save_dir))
        else:
            print("Extracting word2vec data")
            self.init_word()

        try:
            print("Trying to load processed data")
            self.data = pickle.load(open(os.path.join(self.processed_data_dir, self.file_name.split(".")[0] + '.pkl'), 'rb'))
            print("Load successfully")
        except:
            print("Processed data does not exist")
            self._preprocess()

        print("bag num:", self.__len__())


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        bag = self.data[index]
        word = torch.tensor(bag['word'], dtype=torch.long)
        pos1 = torch.tensor(bag['pos1'], dtype=torch.long)
        pos2 = torch.tensor(bag['pos2'], dtype=torch.long)
        ent1 = torch.tensor(bag['ent1'], dtype=torch.long)
        ent2 = torch.tensor(bag['ent2'], dtype=torch.long)
        mask = torch.tensor(bag['mask'], dtype=torch.long)
        length = torch.tensor(bag['length'], dtype=torch.long)
        rel = torch.tensor(bag['rel'], dtype=torch.long)
        que = torch.tensor(bag['query'], dtype=torch.long)
        if self.training:
            rel = rel[0]
        else:
            rel_mul = torch.zeros(len(self.rel2id), dtype=torch.long)
            for i in set(rel):
                rel_mul[i] = 1
            rel = rel_mul
        return word, pos1, pos2, ent1, ent2, mask, length, rel, que


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


    def init_rel(self):
        self.rel2id = json.load(open(self.rel_dir, "r"))


    def rel_num(self):
        return len(self.rel2id)


    def loss_weight(self):
        print("Calculating the class weight")
        rel_ins = []
        for bag in self.data:
            rel_ins.extend(bag['rel'])
        stat = Counter(rel_ins)
        class_weight = torch.ones(self.rel_num(), dtype=torch.float32)
        for k, v in stat.items():
            class_weight[k] = 1. / v ** 0.05
        if torch.cuda.is_available():
            class_weight = class_weight.cuda()
        return class_weight


def collate_fn(X):
    X = list(zip(*X))
    word, pos1, pos2, ent1, ent2, mask, length, rel, que = X

    scope = []
    ind = 0
    for w in word:
        scope.append((ind, ind + len(w)))
        ind += len(w)

    scope_ent1, scope_ent2 = [], []
    ind, ind1, ind2 = 0, 0, 0
    for e1 in ent1:
        scope_ent1.append((ind1, ind1 + len(e1)))
        ind1 += len(e1)
    for e2 in ent2:
        scope_ent2.append((ind2, ind2 + len(e2)))
        ind2 += len(e2)

    scope_ent1 = torch.tensor(scope_ent1, dtype=torch.long)
    scope_ent2 = torch.tensor(scope_ent2, dtype=torch.long)
    scope = torch.tensor(scope, dtype=torch.long)

    word = torch.cat(word, 0)
    pos1 = torch.cat(pos1, 0)
    pos2 = torch.cat(pos2, 0)
    ent1 = torch.cat(ent1, 0)
    ent2 = torch.cat(ent2, 0)
    mask = torch.cat(mask, 0)
    length = torch.cat(length, 0)
    rel = torch.stack(rel)
    que = torch.cat(que, 0)

    return word, pos1, pos2, ent1, ent2, mask, length, scope, scope_ent1, scope_ent2, rel, que


def data_loader(data_file, opt, shuffle, training=True, num_workers=4):
    dataset = Dataset(data_file, opt, training)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt['batch_size'],
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader