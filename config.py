import argparse


def config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data', help='root of data files')

    parser.add_argument('--dataset', default='TBGA', help='Biorel, TBGA')
    parser.add_argument('--save_dir', default='ckpt')
    parser.add_argument('--processed_data_dir', default='_processed_data')

    parser.add_argument('--train', default='train.json')
    parser.add_argument('--dev',   default='dev.json')
    parser.add_argument('--test',  default='test.json')
    parser.add_argument('--rel',   default='relation2id.json')
    parser.add_argument('--vec',   default='word2vec.json')

    parser.add_argument('--encoder', default='BioBERT', type=str, help='PCNN, CNN, BiGRU, BiLSTM or BioBERT')
    parser.add_argument('--epoch',      default=200, type=int, help='Max epochs to train')
    parser.add_argument('--val_iter',   default=1,   type=int)
    parser.add_argument('--early_stop', default=5,  type=int, help='epochs without new best result appearing')

    parser.add_argument('--batch_size',  default=8, type=int)
    parser.add_argument('--lr',          default=0.00001, type=float)
    parser.add_argument('--dropout',     default=0.5, type=float)
    parser.add_argument('--dropout_att', default=0.25, type=float, help='bag_self_att weight dropout')

    parser.add_argument('--max_sentence_length', default=120, type=int)
    parser.add_argument('--max_pos_length',      default=100, type=int)
    parser.add_argument('--max_question_length', default=24, type=int)

    parser.add_argument('--hidden_size', default=230, type=int)
    parser.add_argument('--pos_dim',     default=5,   type=int, help='position embedding dimension')

    parser.add_argument('--bag_size',  default=50,   type=int)

    parser.add_argument('--seed',  default=16051643, type=int)


    return parser.parse_args()