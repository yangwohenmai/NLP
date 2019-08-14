# coding=utf-8
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# 加载词列表
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
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
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

# 对文件夹中每个文件进行过滤处理
def process_docs(directory, vocab):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# 用高频词汇列表vocab，处理每个评论文本
		line = doc_to_line(path, vocab)
		# 将处理后的文本加入列表
		lines.append(line)
	return lines

# load the vocabulary
vocab_filename = r'E:\临时测试程序\pytest\vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# 用词汇列表词汇列表vocab，分别过滤负面正面评论，生成训练数据集合
positive_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\pos', vocab)
negative_lines = process_docs(r'E:\临时测试程序\pytest\txt_sentoken\neg', vocab)
# summarize what we have
print(len(positive_lines), len(negative_lines))