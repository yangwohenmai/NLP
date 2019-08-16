from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

# load doc into memory
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

# 加载训练集/测试集
def process_docs(directory, vocab, is_trian):
	lines = list()
	# 遍历文件夹
	for filename in listdir(directory):
		# 根据参数选择加载的数据集
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# # 用高频词汇列表vocab，处理后的单个评论文本内容
		line = doc_to_line(path, vocab)
		# 将处理后的文本加入列表
		lines.append(line)
	return lines

# 加载词汇频率列表
vocab_filename = r'E:\临时测试程序\pytest\vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# 用词汇列表过滤和拼接训练数据
positive_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\pos', vocab, True)
negative_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\neg', vocab, True)
docs = negative_lines + positive_lines

# 创建一个分词器
tokenizer = Tokenizer()
# 使tokenizer对象识别要编码的数据信息，生成一个编码器
tokenizer.fit_on_texts(docs)

# 将文本编码为矩阵，编码规则为按照词频转换
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtrain.shape)

# 用词汇列表过滤和拼接测试数据
positive_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\pos', vocab, False)
negative_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\neg', vocab, False)
docs = negative_lines + positive_lines
# 将文本编码为矩阵，编码规则为按照词频转换
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtest.shape)