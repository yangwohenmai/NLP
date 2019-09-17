#code=utf-8
from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
"""
本文训练一个简单的Embedding层，如果没有现成可用的嵌入层，就通过训练集数据来训练嵌入层
1.用tokenizer.fit_on_texts()，对训练集文本进行编码，将每个单词对应一个数字索引（单词->数字索引）：tokenizer.word_index
单词->数字索引， 形如：film:1,one:2,movie:3,like:4,or:5
2.用tokenizer.texts_to_sequences()将训练集文本中的句子，转化为对应的数字索引序列：encoded_docs
句子转化为数字序列， 形如：I like one film or movie... -> [?,4,2,1,5,3..] ?代表I对应的数字索引
3.根据最大句长将句子长度对齐
4.创建一个大小为(最大句长*100)的权重矩阵，作为嵌入层
5.构建网络结构：嵌入->卷积->池化->平滑->分类
6.将数字序列化后的训练数据，带入网络进行训练，反向传播更新嵌入层权重矩阵
7.评估网络性能
"""

# 加载文件
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
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
    #用空格链接单词
	tokens = ' '.join(tokens)
	return tokens

# 文本清洗
def process_docs(directory, vocab, is_trian):
	documents = list()
	# 遍历文件夹下文本文件
	for filename in listdir(directory):
		# 根据is_trian参数选取训练集/测试集
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# 加载评论文本
		doc = load_doc(path)
		# 对文本进行清洗和格式化
		tokens = clean_doc(doc, vocab)
		# 将处理后的评论文本加入列表
		documents.append(tokens)
	return documents


# 加载生成好的词汇列表vocab
vocab_filename = r'E:\MyGit\NLP\文本分类\实例_基于word2vec词向量分类\vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# 用词汇列表过滤和拼接 训练 数据
positive_docs = process_docs(r'E:\MyGit\NLP\文本分类\实例_基于word2vec词向量分类\txt_sentoken/pos', vocab, True)
negative_docs = process_docs(r'E:\MyGit\NLP\文本分类\实例_基于word2vec词向量分类\txt_sentoken/neg', vocab, True)
# 将处理过的(正面/负面)训练数据集组合起来
train_docs = negative_docs + positive_docs

# 创建一个分词器
tokenizer = Tokenizer()
# 使tokenizer对象对于文本中每个单词进行编码，每个单词对应一个数字
tokenizer.fit_on_texts(train_docs)

# sequence encode对文本单词进行索引编码，把文本中每一个单词对应一个整数索引
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# 获取列表中最大句子长度
max_length = max([len(s.split()) for s in train_docs])
# 对编码后的文本列表中不同长度的句子，用0填充到相同长度
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# 测试集前900负面数据标记位0，后900正面数据标记位1
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])


# 用词汇列表过滤和拼接 测试 数据
positive_docs = process_docs(r'E:\MyGit\NLP\文本分类\实例_基于word2vec词向量分类\txt_sentoken/pos', vocab, False)
negative_docs = process_docs(r'E:\MyGit\NLP\文本分类\实例_基于word2vec词向量分类\txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs

# sequence encode对文本单词进行索引编码，把文本中每一个单词对应一个整数索引
# 使用训练集的tokenizer分词器，来编码测试集文本
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# 对编码后的文本列表中不同长度的句子，用0填充到相同长度
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# 测试集前100负面数据标记位0，后100正面数据标记位1
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# 定义词汇量大小=训练集总词汇量+1
vocab_size = len(tokenizer.word_index) + 1

# 定义网络
model = Sequential()
# 词嵌入层作为第一个隐藏层。参数：词汇量大小，每个word向量为100维，输入文本最大长度
model.add(Embedding(vocab_size, 100, input_length=max_length))
# 使用一维CNN卷积神经网络，32个滤波器，激活函数为relu，卷积核数量为4
# 一维卷积一般会处理时序数据，所以，卷积核的宽度为1，kernel_size是卷积核的长度
model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
# 池化层
model.add(MaxPooling1D(pool_size=2))
# 将CNN输出的2D输出展平为一个长2D矢量，以表示由CNN提取的“特征”
model.add(Flatten())
# 整流全连接层，标准的多层感知器层，用于解释CNN功能
model.add(Dense(10, activation='relu'))
# sigmoid分类函数输出分类预测
model.add(Dense(1, activation='sigmoid'))
# 打印模型结构
print(model.summary())
# 损失函数：二元交叉熵，梯度优化器：adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练网络，10轮，每轮输出日志
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# 用测试集评估网络准确度
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))

# 调用训练的模型进行分类测试
# test positive text
text = 'best movie ever! i like it very much, haha'
line = clean_doc(text,vocab)
encoded_docs = tokenizer.texts_to_sequences([line])
Xhat = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
yhat = model.predict(Xhat,verbose=2)
print(round(yhat[0,0]))
# test negative text
text = 'this is a bad movie.i dont want see it again'
line = clean_doc(text,vocab)
encoded_docs = tokenizer.texts_to_sequences([line])
Xhat = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
yhat = model.predict(Xhat,verbose=2)
print(round(yhat[0,0]))