"""Utilities for training the dependency parser.
You do not need to read/understand this code
"""

import time
import os
import logging
from collections import Counter
from general_utils import logged_loop, get_minibatches
from q2_parser_transitions import PartialParse, minibatch_parse

import numpy as np


P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'  # 不存在于字典中的字符，比如只出现一次的词
NULL = '<NULL>'  # 空字符
ROOT = '<ROOT>'


class Config(object):
    language = 'english'
    with_punct = True  # 是否要将符号也解析
    unlabeled = False  # label 除了 S LA RA 外，是否要在后面加上依赖，如 S-NN
    lowercase = True
    use_pos = True  # 特征构造时是否要用细粒度词性
    use_dep = True
    use_dep = use_dep and (not unlabeled)  # 特征构造时是否要用依赖关系，必须label有依赖关系才行，即unlabeled=False
    data_path = './data'
    train_file = 'train.conll'
    dev_file = 'dev.conll'
    test_file = 'test.conll'
    embedding_file = './data/en-cw.txt'


class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""

    # # train_set: [{word:[..], pos:[..], head:[..], label:[..]},{..},..]，一个字典表示一个句子
    def __init__(self, dataset):
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)
        if len(counter) > 1:  # 如果 root 还有其他的表示，如大写的 ROOT 等
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]  # 取出最多的一种表示方法
        deprel = [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))  # 取出所有依赖关系
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)

        config = Config()
        self.unlabeled = config.unlabeled
        self.with_punct = config.with_punct
        self.use_pos = config.use_pos
        self.use_dep = config.use_dep
        self.language = config.language

        if self.unlabeled:
            trans = ['L', 'R', 'S']
            self.n_deprel = 1  # 用于后面方便判断
        else:
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)

        self.n_trans = len(trans)
        self.tran2id = {t: i for (i, t) in enumerate(trans)}
        self.id2tran = {i: t for (i, t) in enumerate(trans)}

        # logging.info('Build dictionary for part-of-speech tags.')
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                  offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)  # P_UNK 的id
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)  # P_NULL 的id
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)  # P_ROOT 的id

        # logging.info('Build dictionary for words.')
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                  offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)  # UNK 的id
        tok2id[NULL] = self.NULL = len(tok2id)  # NULL 的id
        tok2id[ROOT] = self.ROOT = len(tok2id)  # ROOT 的 id

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()}

        self.n_features = 18 + (18 if config.use_pos else 0) + (12 if config.use_dep else 0)
        self.n_tokens = len(tok2id)

    def vectorize(self, examples):
        vec_examples = []
        for ex in examples:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]  # word首位是root的id..
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]
            head = [-1] + ex['head']
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})
        return vec_examples


    def extract_features(self, stack, buf, arcs, ex):
        ''' feature构造方式: 6 (stack和buf头三个word) + 12 (stack顶2个词的左右最远的两个依赖以及最远依赖的最远依赖)
                  use_pos: +6(上面的pos)             + 12 (上面的pos, 细粒度词性)
                  use_dep:                          + 12 (上面的dep，依赖关系类型)
        '''
        if stack[0] == "ROOT":
            stack[0] = 0

        def get_lc(k):  # 获得左独立项id,箭头指向左边的那些,从小（最远的）到大
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):  # 获得右独立项id,箭头指向右边的那些，从大（最远的）到小
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        p_features = []
        l_features = []
        features = [self.NULL] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]]  # stack顶3位元素id
        features += [ex['word'][x] for x in buf[:3]] + [self.NULL] * (3 - len(buf))  # buf顶3位元素id
        if self.use_pos:
            p_features = [self.P_NULL] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]]
            p_features += [ex['pos'][x] for x in buf[:3]] + [self.P_NULL] * (3 - len(buf))

        for i in range(2):  # 遍历stack顶两位
            if i < len(stack):
                k = stack[-i-1]
                lc = get_lc(k)
                rc = get_rc(k)
                llc = get_lc(lc[0]) if len(lc) > 0 else []
                rrc = get_rc(rc[0]) if len(rc) > 0 else []

                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.NULL)
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.NULL)

                if self.use_pos:
                    p_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.P_NULL)

                if self.use_dep:
                    l_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.L_NULL)
            else:
                features += [self.NULL] * 6
                if self.use_pos:
                    p_features += [self.P_NULL] * 6
                if self.use_dep:
                    l_features += [self.L_NULL] * 6

        features += p_features + l_features
        assert len(features) == self.n_features
        return features

    # 根据example内容，返回当前state所执行的正确操作
    def get_oracle(self, stack, buf, ex):
        if len(stack) < 2:  # 如果stack上面只有root，就执行shift，返回对应tran的编号
            return self.n_trans - 1

        i0 = stack[-1]
        i1 = stack[-2]
        h0 = ex['head'][i0]
        h1 = ex['head'][i1]
        l0 = ex['label'][i0]
        l1 = ex['label'][i1]

        if self.unlabeled:
            if (i1 > 0) and (h1 == i0):
                return 0
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return 1
            else:
                return None if len(buf) == 0 else 2
        else:
            if (i1 > 0) and (h1 == i0):
                return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
            else:
                return None if len(buf) == 0 else self.n_trans - 1

    def create_instances(self, examples):
        '''
        :param examples: [{word:[..], pos:[..], head:[..], label:[..]}..]，一个字典表示一个句子，里面用token2id的id表示
        :return: [[([n_feature长的特征], [可以采取的所有操作], 真实的操作)..]..]
        '''
        all_instances = []
        succ = 0
        for id, ex in enumerate(logged_loop(examples)):
            n_words = len(ex['word']) - 1

            # arcs = {(h, t, label)}
            stack = [0]
            buf = [i + 1 for i in range(n_words)]
            arcs = []
            instances = []
            for i in range(n_words * 2):  # 每个词有进有出，一共有2n步
                gold_t = self.get_oracle(stack, buf, ex)
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))
                if gold_t == self.n_trans - 1:
                    stack.append(buf[0])
                    buf = buf[1:]
                elif gold_t < self.n_deprel:  # 说明是LA-XXX
                    arcs.append((stack[-1], stack[-2], gold_t))
                    stack = stack[:-2] + [stack[-1]]
                else:
                    arcs.append((stack[-2], stack[-1], gold_t - self.n_deprel))
                    stack = stack[:-1]
            else:
                succ += 1
                all_instances += instances

        return all_instances

    # 当前可以允许的所有移位操作
    def legal_labels(self, stack, buf):
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel  # 所有类型的左移操作
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel  # 所有类型的右移操作
        labels += [1] if len(buf) > 0 else [0]  # shift操作
        return labels

    # 返回 UAS 匹配所占百分比，以及用NN模型对dataset进行解析后的依赖
    def parse(self, dataset, eval_batch_size=5000):
        sentences = []
        sentence_id_to_idx = {}
        for i, example in enumerate(dataset):
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i

        model = ModelWrapper(self, dataset, sentence_id_to_idx)
        dependencies = minibatch_parse(sentences, model, eval_batch_size)

        UAS = all_tokens = 0.0
        for i, ex in enumerate(dataset):
            head = [-1] * len(ex['word'])
            for h, t, in dependencies[i]:
                head[t] = h
            for pred_h, gold_h, gold_l, pos in \
                    zip(head[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
                    assert self.id2tok[pos].startswith(P_PREFIX)
                    pos_str = self.id2tok[pos][len(P_PREFIX):]
                    if (self.with_punct) or (not punct(self.language, pos_str)):
                        UAS += 1 if pred_h == gold_h else 0
                        all_tokens += 1
        UAS /= all_tokens
        return UAS, dependencies


# 只是一个封装，用的还是parser里面的model来解析
class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx

    def predict(self, partial_parses):
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies,
                                             self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32')  # list 转array
        mb_l = [self.parser.legal_labels(p.stack, p.buffer) for p in partial_parses]
        pred = self.parser.model.predict_on_batch(self.parser.session, mb_x)
        pred = np.argmax(pred + 10000 * np.array(mb_l).astype('float32'),
                         1)  # 这里10000表示一个是预测出来的操作，一个是通过判断绝对不可能的操作，首先要满足那些绝对不可能的操作的权重很低，反过来就是可能的操作权重高
        pred = [self.parser.id2tran[p] for p in pred]
        return pred


# word: 词
# pos: 细粒度词性
# head: 中心词 ： 谁指向它 = 修饰哪个中心词
# label: 依存关系
# max_example : 最多解析前几个句子
def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:  # 遇到换行，表示一个句子已经结束，append
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    return examples


def build_dict(keys, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)

    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def punct(language, pos):
    if language == 'english':
        return pos in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    elif language == 'chinese':
        return pos == 'PU'
    elif language == 'french':
        return pos == 'PUNC'
    elif language == 'german':
        return pos in ["$.", "$,", "$["]
    elif language == 'spanish':
        # http://nlp.stanford.edu/software/spanish-faq.shtml
        return pos in ["f0", "faa", "fat", "fc", "fd", "fe", "fg", "fh",
                       "fia", "fit", "fp", "fpa", "fpt", "fs", "ft",
                       "fx", "fz"]
    elif language == 'universal':
        return pos == 'PUNCT'
    else:
        raise ValueError('language: %s is not supported.' % language)


def minibatches(data, batch_size):
    '''

    :param data: [([n_feature长的特征], [0 1，1表示可以采取的操作], 真实的操作)..]
    :param batch_size: ..
    :return: one-hot编码真实操作作为label
    '''
    x = np.array([d[0] for d in data])
    y = np.array([d[2] for d in data])
    one_hot = np.zeros((y.size, len(data[0][1])))
    one_hot[np.arange(y.size), y] = 1
    return get_minibatches([x, one_hot], batch_size)


def load_and_preprocess_data(reduced=True):
    config = Config()

    # 1. 加载三部分conll格式的数据
    print("Loading data...", )
    start = time.time()
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)
    test_set = read_conll(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]
    print("took {:.2f} seconds".format(time.time() - start))

    # 2. 初始化parser, 构建token2id, id2token等
    print("Building parser...", )
    start = time.time()
    parser = Parser(train_set)
    print("took {:.2f} seconds".format(time.time() - start))

    # 3. 加载提前训练好的词向量，遍历训练数据的POS, label, word的id, 如果有预训练词向量，就放到词嵌入矩阵里，否则就是随机的
    print("Loading pretrained embeddings...", )
    start = time.time()
    word_vectors = {}
    for line in open(config.embedding_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')

    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
    print("took {:.2f} seconds".format(time.time() - start))

    # 4. 将train_set..里面都用token2id的id表示，后面两种里面会用到UKN的word，因为token2id是根据trainset构建的
    print("Vectorizing data...", )
    start = time.time()
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Preprocessing training data...")
    train_examples = parser.create_instances(train_set)

    return parser, embeddings_matrix, train_examples, dev_set, test_set,

if __name__ == '__main__':
    load_and_preprocess_data()
