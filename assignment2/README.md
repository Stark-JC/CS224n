[toc]

# 前言
本篇主要对CS224N的assignment2中的question2代码进行理解分析。
# Parser
一个parser需要以下几个部分：
1. 解析器全局参数
    -  `unlabeled`: label 除了 S, LA, RA 外，是否要在后面加上依赖关系，如 S-NN
    - `use_dep` : 特征构造时是否要用依赖关系，必须label有依赖关系才行，即unlabeled=False
    - `use_pos` : 特征构造时是否要用细粒度词性
    - `with_punct`: 是否要将符号也解析
    - ..
2. id映射
    - `id2token, token2id` : 所有word, pos, dep与id的映射，id用于表示输入X，以及embeding lookup
    - `tran2id, id2tran` : 所有转移操作与id的映射关系，id用于表示输出y
3. ParserModel: 用于预测某个状态的决策。可用tensorflow搭建。输入是`n_features`维，输出`n_classes`维

## Parser 类
### `__init__(self, dataset)`
输入是从conll格式文件读来的训练数据，格式为
```
[{word:[..], pos:[..], head:[..], label:[..]},{..},..]
```
，一个字典表示一个句子，一个句子里面有词，对应词性，对应依赖关系的head, 以及对应依赖关系的类型，这里输入都是原始的字符串值，然后在init里面进行2token的转换。

### `vectorize(self, examples)`
对输入字符串的example转换为对应id表示的example, 这里example的数据格式和上面的dataset的数据格式一致。

### `create_instances(self, examples)`
创建格式化的训练数据。
```python
'''
:param examples: [{word:[..], pos:[..], head:[..], label:[..]}..]，一个字典表示一个句子，里面用token2id的id表示，即vectorize后的。
:return: [[([n_feature长的特征], [可以采取的操作], 真实的操作)..]..]
'''
```
具体操作为：
1. 对一个example, 初始化stack, buf, arcs.
2. 循环2*n_word次:
    1. 调用`get_oracle`获得本步应该执行的正确tran;
    2. 调用`legal_labels`获得本步允许执行的所有tran;
    3. 调用`extract_features`获得本步提取出来的特征；
    4. 上面三个拼装成一个tuple,作为一步的所有信息。
    5. 根据`get_oracle`返回的正确的操作调整stack, buf, arcs。
3. 所有步骤合成的list加到大list中。

由于该训练数据还不是纯种的(X,y)形式，故需要有个函数将这些数据转换成标准的格式，代码中是`minibatch(data, batch_size)`函数，用与对data进行转换（主要是y的格式变成one-hot），然后返回一个minibatch生成器。

#### `extract_features(self, stack, buf, arcs, ex)`
根据config里面指定的feature构造方式，对当前状态进行feature构造，stack, buf， arcs里面都是针对该句子的id，如首位就是id=1..
返回的是转换后的全局token2id的id

#### `get_oracle(self, stack, buf, ex)`
根据example内容，返回当前state所执行的正确操作，因为example里面有head。主要比较stack顶两位对应head之前的关系。具体思路如下：

```python
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
```
#### `legal_labels(self, stack, buf)`
返回当前可以允许的所有移位操作的id。

### `parse(self, dataset, eval_batch_size)`
dataset是id化的形式。
返回预测head list与正确head list匹配所占百分比（UAS），以及用NN模型对dataset进行解析后的依赖。
该函数用于dev集以及test集UAS的计算，训练集不会用到这个，因为训练集不用UAS作为误差， 而是cross-entropy.

调用了一个minibatch_parse来得到预测的依赖。

#### `minibatch_parse(sentences, model, batch_size)`
1. 对每个sentence构造一个`PartialParse`（这个`PartialParse`是没有预测功能的，里面没有model，只维护stack, buf, arcs以及移位操作，需要由外界提供transitions）
2. 每次预测minibatch_size大小的sentence得到transitions， 这里预测是通过传入的model实现的，得到的transitions是字符串的移位操作。
3. 得到了预测的transitions，传入到PartialParse的parse_step函数进行dependencies的更新。

这里我们又发现model的预测不是softmax之后的概率向量吗？怎么直接得到字符串形式的移位操作？其实也很简单，代码中对原本的ParserModel的predict函数进行了封装，用了一个ModelWrapper类。
##### ModelWrapper 类
代码中对于预测结果，不单单依据神经网络预测的概率直接进行取最大值操作，还考虑到了当前的允许执行的操作（例如第一步预测出来是LA，但我们知道这个预测100%是错的），就是要事先得排除那些不可能的trans, 代码中采用可能操作权重*10000的操作。
```python
# 只是一个封装，用的还是parser里面的model来解析
class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx
    
    def predict(self, partial_parses):
        # 根据PartialParser所维护的状态，得到该状态下的feature表示。
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies, self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32')  # list 转array
        
        # 根据PartialParser所维护的状态，得到该状态下允许的label表示。
        mb_l = [self.parser.legal_labels(p.stack, p.buffer) for p in partial_parses]
        
        # 这个pred得到的是每个移位操作的概率
        pred = self.parser.model.predict_on_batch(self.parser.session, mb_x)
        
        # 这里10000表示一个是预测出来的操作，一个是通过判断绝对不可能的操作，首先要满足那些绝对不可能的操作的权重很低，反过来就是可能的操作权重高
        pred = np.argmax(pred + 10000 * np.array(mb_l).astype('float32'),
                         1)  
        pred = [self.parser.id2tran[p] for p in pred]
        
        # 返回处理后的预测的trans字符串
        return pred
```

## ParserModel 类
用于预测某个状态的决策。可用tensorflow搭建。

输入是某一个state（stack, buf, arc）的特征的向量表示，例如一种36维的构造方式为：
```
feature构造方式: 6 (stack和buf头三个word) + 12 (stack顶2个词的左右最远的两个依赖词以及最远依赖词的最远依赖词)
      use_pos: +6(上面的pos)             + 12 (上面的pos, 细粒度词性)
```
这样，输入就是36维的向量，每维的值都是事先建立索引的token2id里面的id号，一般在第二层是lookup层，查找id的预训练词向量。

输出是在改state的决策，如上面这种特征构造方式，决策有三类：LA, RA, S。