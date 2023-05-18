import json
import numpy as np
import re
import os

patten = r"{\"text\": \"(.+)\", \"relation\": \"(.+)\", \"h\": {\"id\": (.+), \"name\": \"(.+)\", \"pos\": \[(.+), (.+)\]}, \"t\": {\"id\": (.+), \"name\": \"(.+)\", \"pos\": \[(.+), (.+)\]}}"
dataset_dir = 'TBGA'

if not os.path.exists(dataset_dir):
    raise IndexError('Dataset TBGA not exist, please check file')

for cur_dataset in ['train', 'test', 'val']:
    dealed_data = []
    with open('TBGA_' + cur_dataset + '.txt') as f:
        for line in f.readlines():
            ins = {
                "sentence": '',
                "relation": 'NA',
                "head": {},
                "tail": {},
            }
            tmp = re.findall(patten, line)[0]

            ins["sentence"] = tmp[0].lower()
            ins["relation"] = tmp[1]

            if ", " in ins["sentence"]:
                ins["sentence"] = ins["sentence"].replace(", ", " , ")
            if ins["sentence"][-1] == '.':
                ins["sentence"] = ins["sentence"][:-1] + " ."

            sentence = ins["sentence"].split()

            ent1, ent2 = tmp[3].lower(), tmp[7].lower()
            ent1_, ent2_ = ent1.split(), ent2.split()
            if len(ent1_) > 1:
                head_spilt_start = sentence.index(ent1_[0]) if ent1_[0] in sentence else 0
            else:
                head_spilt_start = sentence.index(ent1) if ent1 in sentence else 0
            if len(ent2_) > 1:
                tail_spilt_start = sentence.index(ent2_[0]) if ent2_[0] in sentence else 0
            else:
                tail_spilt_start = sentence.index(ent2) if ent2 in sentence else 0

            head_start = len(' '.join(sentence[:head_spilt_start])) if head_spilt_start == 0 else len(' '.join(sentence[:head_spilt_start])) + 1
            tail_start = len(' '.join(sentence[:tail_spilt_start])) if tail_spilt_start == 0 else len(' '.join(sentence[:tail_spilt_start])) + 1


            ins["head"]["CUI"] = tmp[2].replace("\"", '')
            ins["head"]["word"] = ent1
            ins["head"]["start"] = head_start
            ins["head"]["length"] = len(ent1)
            ins["head"]["split_start"] = head_spilt_start

            ins["tail"]["CUI"] = tmp[6].replace("\"", '')
            ins["tail"]["word"] = ent2
            ins["tail"]["start"] = tail_start
            ins["tail"]["length"] = len(ent2)
            ins["tail"]["split_start"] = tail_spilt_start

            dealed_data.append(ins)

    if cur_dataset == 'val': # Align valid dataset format to dev.json
        cur_dataset = 'dev'

    with open(os.path.join(dataset_dir, cur_dataset + '.json'), 'w') as f:
        json_str = json.dumps(dealed_data, indent='\t')
        f.write(json_str)
        f.write('\n')



if not os.path.exists(os.path.join(dataset_dir, 'biowordvec_word2id.json')):
    raise IndexError('Dataset TBGA not exist, please check file')
index = json.load(open(os.path.join(dataset_dir, 'biowordvec_word2id.json'), 'r'))
vec = json.load(open(os.path.join(dataset_dir, 'biowordvec_mat.npy'), 'r'))
# biowordvec_word2id.json and biowordvec_mat.npy can be obtained at https://github.com/GDAMining/gda-extraction
# we use biowordvec as word embeddings for TBGA dataset

with open(os.path.join(dataset_dir, 'word2vec.json'), 'w') as f:
    word2vec = []
    for word in index.keys():
        cur_vec = {}
        cur_vec["word"] = word
        tmp = [str(_) for _ in list(vec[index[word]])]
        cur_vec["vec"] = [float(_) for _ in tmp]
        word2vec.append(cur_vec)
    json.dump(word2vec, f)



with open(os.path.join(dataset_dir, 'relation2id'), 'w') as f:
    rel = json.load(open(os.path.join(dataset_dir, 'TBGA_rel2id.json'), 'r'))
    json.dump(rel, f)