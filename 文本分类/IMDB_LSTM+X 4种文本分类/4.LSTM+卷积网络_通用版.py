from string import punctuation
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from os import listdir
from numpy import array
"""
本文从IMDB上获取训练好的单词索引，训练文本数据也是已经转换成数字序列。
因此本文不具有通用性，自己的文本不能直接用来训练和测试，因此将程序加以改造。
1.首先，通过imdb.get_word_index()，来加载IMDB官方的“单词-索引”字典
2.将训练文本通过IMDB的“单词-索引”字典，转换成对应的数字序列
3.利用网络模型进行分类
"""

# fix random seed for reproducibility
numpy.random.seed(7)
# 加载IMDB数据，只加载前top_words个经常使用的词汇（下载的训练文本中只包含前top_words的词汇）
top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# imdb.get_word_index()获取的是IMDB官方的“单词-索引”字典
word_index = imdb.get_word_index()
# 处理一些特殊字符，对自己的文本转换可以不考虑
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0  # 没有出现但字典里有的词
word_index["<START>"] = 1# 起始符号
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# 获取“单词-数字”对应字典
change_word_index = dict([(key, value) for (key, value) in word_index.items()])
# 获取“数字-单词”对应字典
# change_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 将文本转化为数字
def text_to_index(text, change_word_index):
    try:
        ll = list()
        # ll.append(1)
        # 转换文本时，只保存imdb.get_word_index()中出现频率为前top_words的单词
        ll.extend([change_word_index.get(word,0) for word in text if change_word_index.get(word,0) < top_words])
        #llarray = numpy.asarray(ll, dtype = int)
        #string = ' '.join('%s' %id for id in ll)
    except Exception as e:
        print(e)
    return ll


# 加载文件
def load_doc(filename):
	file = open(filename, 'r',encoding="utf-8")
	text = file.read()
	file.close()
	return text

# 对文本进行清洗和格式化方法
def clean_doc(doc, vocab):
	# 按照空格分词
	tokens = doc.split()
	# 删除文本中的标点符号
    # table是生成的字符映射规则，规则为将标点符号映射为空字符->''
	table = str.maketrans('', '', punctuation)
    # 根据生成的映射规则table，对tokens中每个单词进行转义映射
	tokens = [w.translate(table) for w in tokens]
	# 使用统计的高频词列表vocab，过滤文本中的低频单词
	tokens = [w for w in tokens if w in vocab]
	return tokens

# 文本清洗，转化成数字序列
def process_docs(directory, is_trian):
    documents = list()
    # 遍历文件夹下文本文件
    for filename in listdir(directory):
        # 根据is_trian参数选取训练集/测试集
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        # 加载评论文本
        doc = load_doc(path)
        # 对文本进行清洗和格式化
        tokens = clean_doc(doc, vocab)
        # 将文本转化为数字序列，list存放
        tokens_index = text_to_index(tokens, change_word_index)
        # 将转换后的所有评论放入数组
        documents.append(tokens_index)
    return numpy.asarray(documents)


# 加载生成好的词汇列表vocab
vocab_filename = r'E:\MyGit\NLP\文本分类\IMDB_LSTM+X 4种文本分类\vocab2.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


# 根据IMDB的“单词-索引”字典，将训练文本转换为数字索引
xtrain_pos = process_docs(r'E:\MyGit\NLP\文本分类\IMDB_LSTM+X 4种文本分类\txt_sentoken/pos', True)
xtrain_neg = process_docs(r'E:\MyGit\NLP\文本分类\IMDB_LSTM+X 4种文本分类\txt_sentoken/neg', True)
# 训练数据合并
xtrain = numpy.concatenate((xtrain_neg,xtrain_pos),axis=0)
# 制造评价数据
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# 准备测试数据
xtest_pos = process_docs(r'E:\MyGit\NLP\文本分类\IMDB_LSTM+X 4种文本分类\txt_sentoken/pos', False)
xtest_neg = process_docs(r'E:\MyGit\NLP\文本分类\IMDB_LSTM+X 4种文本分类\txt_sentoken/neg', False)
xtest = numpy.concatenate((xtest_neg,xtest_pos),axis=0)
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# 每个句子固定长度为500
max_review_length = 500
# 对齐训练数据
xtrain = sequence.pad_sequences(xtrain, maxlen=max_review_length)
# 对齐测试数据
xtest = sequence.pad_sequences(xtest, maxlen=max_review_length)

# 词向量维度
embedding_vecor_length = 32
model = Sequential()
# 嵌入层 参数：词汇数量，句中每个单词的向量维度32，每条评论句长500
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 一维卷积提取特征，32个卷积核，取词窗口为3，补零策略same（查阅kers笔记），整流激活函数
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# 池化层
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
# 分类输出
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# 训练网络
model.fit(xtrain, ytrain, epochs=15, batch_size=64)
# 评估网络准确度
scores = model.evaluate(xtest, ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))