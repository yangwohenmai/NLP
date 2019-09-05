from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

"""
本文内容：
1.用tokenizer.fit_on_texts()，对训练集文本进行编码，将每个单词对应一个数字索引（单词->数字索引）：tokenizer.word_index
单词->数字索引， 形如：film:1,one:2,movie:3,like:4,or:5
2.用tokenizer.texts_to_sequences()将训练集文本中的句子，转化为对应的数字索引序列：encoded_docs
句子转化为数字序列， 形如：I like one film or movie... -> [?,4,2,1,5,3..] ?代表I对应的数字索引
3.利用训练集文本生成的词向量word2vec文件，生成一个(单词->向量)映射关系字典：raw_embedding
4.利用映射关系字典raw_embedding，为tokenizer.word_index中的单词，匹配其所对应的向量
film:[11,22,33..] 
one:[22,33,44..] 
movie:[33,44,55..]
5.生成嵌入层权重矩阵：
以tokenizer.word_index中单词对应的索引，作为权重矩阵的行顺序，填充矩阵
tokenizer.word_index索引为：
film:1,
one:2,
movie:3....
生成的权重矩阵为:
[
[11,22,33..],
[22,33,44..],
[33,44,55..]
...]
6.将权重矩阵构建成嵌入层结构：embedding_layer
7.构建网络模型：嵌入层->卷积层->池化层->平滑层->分类层
8.设置损失函数等，训练网络等，评估网络准确度等

本文重点：
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# 用训练集文本生成的权重矩阵，构建 词嵌入层embedding_layer，trainable=False表示训练过程中不对嵌入层权重进行调整

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)
"""

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

# 将word2vec构建成(单词->向量)的映射关系字典
def load_embedding(filename):
	file = open(filename,'r')
    # 跳过第一行
	lines = file.readlines()[1:]
	file.close()
	# 构建一个(单词->向量)的映射关系字典
	embedding = dict()
	for line in lines:
		parts = line.split()
		# 将word2vec文件构建成字典，key：单词, value：单词所对应的向量 组成数组
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

# 根据word2vec中(单词->向量)的映射关系，将训练集中文本，转化为一个嵌入层权重矩阵
def get_weight_matrix(embedding, train_text_index):
	# 词汇量+1
	vocab_size = len(train_text_index) + 1
	# 为训练集文本中每个单词创建一个0向量矩阵，每个向量为100维,组成一个vocab_size*100的矩阵
    # 形如：[[0,0...,0,0],[0,0...,0,0],...[0,0...,0,0]]
	weight_matrix = zeros((vocab_size, 100))
	# 根据word2vec文件的(单词->向量)映射关系字典,将训练集的文本 构建成 嵌入层的权重矩阵
	for word, i in train_text_index.items():
        #获取训练集文本中的单词 在word2vec文件中对应的向量，赋值给矩阵中的某一行
		weight_matrix[i] = embedding.get(word)
	return weight_matrix

# 加载生成好的词汇列表vocab
vocab_filename = r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# 用词汇列表过滤和拼接 训练 数据
positive_docs = process_docs(r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\txt_sentoken/pos', vocab, True)
negative_docs = process_docs(r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\txt_sentoken/neg', vocab, True)
# 将处理过的(正面/负面)训练数据集组合起来
train_docs = negative_docs + positive_docs

# 创建一个分词器
tokenizer = Tokenizer()
# 使tokenizer对象对于文本中每个单词进行编码，每个单词对应一个数字
tokenizer.fit_on_texts(train_docs)

# 对文本单词进行索引编码，把文本中每一个单词转化成一个整数，文本变成了数字序列
# 12,3,23,45,23,45
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# 获取列表中最大句子长度
max_length = max([len(s.split()) for s in train_docs])
# 对编码后的文本-数字列表中不同长度的句子，用0填充到相同长度
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# 测试集前900负面数据标记位0，后900正面数据标记位1
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# 用词汇列表过滤和拼接 测试 数据
positive_docs = process_docs(r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\txt_sentoken/pos', vocab, False)
negative_docs = process_docs(r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs

# 对文本单词进行索引编码，把文本中每一个单词转化成一个整数，文本变成了数字序列
# 句子变成了数字序列：12,3,23,45,23,45
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# 对编码后的文本列表中不同长度的句子，用0填充到相同长度
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# 测试集前100负面数据标记位0，后100正面数据标记位1
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# 定义词汇量大小=训练集总词汇量+1
vocab_size = len(tokenizer.word_index) + 1

# 加载训练好的词嵌入文件word2vec.txt，生成一个(单词->向量)映射关系字典
raw_embedding = load_embedding(r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\embedding_word2vec.txt')
# 通过使用word2vec文件生成的(单词->向量)影射关系字典，与训练集文本中的单词 匹配其对应的向量，将训练集文本 转化成一个由向量构成的 权重矩阵
# raw_embedding是训练好的词向量，tokenizer.word_index是带转换的文本
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# 用训练集文本生成的权重矩阵，构建 词嵌入层embedding_layer，trainable=False表示训练过程中不对嵌入层权重进行调整
# embedding_vectors是通过训练好的词向量转换成的文本
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

# 定义训练模型
model = Sequential()
# 词嵌入层作为第一层
model.add(embedding_layer)
# 作为特征提取的卷积层
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
# 池化层
model.add(MaxPooling1D(pool_size=2))
# 平滑层，将多维数据转化为一维数据，用于卷积层到全连接层的过度
model.add(Flatten())
# 分类层
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# 设置损失函数，梯度优化器，评估函数用 准确度
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))