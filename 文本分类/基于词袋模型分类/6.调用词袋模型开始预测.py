from numpy import array
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# 加载文本文件
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# 清理文本，转化成词列表
def clean_doc(doc):
	# 按照空格分词
	tokens = doc.split()
	# 删除文本中的标点符号
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# 删除非字母字符
	tokens = [word for word in tokens if word.isalpha()]
	# 删除已知的停用词
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# 删除单个字母
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# 加载文本, 清洗文本，过滤文本单词，用空格链接生成字符串
def doc_to_line(filename, vocab):
	# 加载单个评论文本
	doc = load_doc(filename)
	# 清理文本，转化成词列表
	tokens = clean_doc(doc)
	# 使用统计的高频词列表，过滤文本中的低频单词
	tokens = [w for w in tokens if w in vocab]
    #用空格链接单词
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# 遍历文件夹下文本文件
	for filename in listdir(directory):
		# 根据is_trian参数选取训练集/测试集
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# 用高频词汇列表vocab，处理每个评论文本
		line = doc_to_line(path, vocab)
		# 将处理后的文本加入列表
		lines.append(line)
	return lines
    
    
    
# 调用模型开始预测，1代表正面，0代表负面
def predict_sentiment(review, vocab, tokenizer, model):
	# clean
	tokens = clean_doc(review)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	# convert to line
	line = ' '.join(tokens)
	# encode
	encoded = tokenizer.texts_to_matrix([line], mode='freq')
	# prediction
	yhat = model.predict(encoded, verbose=0)
	return round(yhat[0,0])

# 加载生成好的词汇列表
vocab_filename = r'E:\临时测试程序\pytest\vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# 用词汇列表过滤和拼接 训练 数据
positive_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\pos', vocab, True)
negative_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\neg', vocab, True)
# create the tokenizer
tokenizer = Tokenizer()
# 将处理过的训练数据集组合起来
docs = negative_lines + positive_lines
# 使tokenizer对象识别要编码的数据类型
tokenizer.fit_on_texts(docs)

# 将训练文本编码为矩阵，编码规则为按照词频编码
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
# 测试集前900负面数据标记位0，后900正面数据标记位1
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# 用词汇列表过滤和拼接 测试 数据
positive_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\pos', vocab, False)
negative_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\neg', vocab, False)
docs = negative_lines + positive_lines
# 将测试文本编码为矩阵，编码规则为按照词频编码
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
# 测试集前100负面数据标记位0，后100正面数据标记位1
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
# 每个词向量的维数
n_words = Xtest.shape[1]
# 定义网络
model = Sequential()
model.add(Dense(50, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 二元交叉熵损失函数 Adam随机梯度下降模型 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 开始训练网络
model.fit(Xtrain, ytrain, epochs=50, verbose=2)
# 模型预测评估，计算在测试集上的准确率
#loss, acc = model.evaluate(Xtest, ytest, verbose=0)
#print('Test Accuracy: %f' % (acc*100))

# test positive text
text = 'Best movie ever!'
print(predict_sentiment(text, vocab, tokenizer, model))
# test negative text
text = 'This is a bad movie.'
print(predict_sentiment(text, vocab, tokenizer, model))


