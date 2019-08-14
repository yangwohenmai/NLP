# coding=utf-8
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# 加载文件
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

# 加载文件，生成词列表，计数
def add_doc_to_vocab(filename, vocab):
	# 加载文本
	doc = load_doc(filename)
	# 清理文本，转化成词列表
	tokens = clean_doc(doc)
	# 更新词列表计数器
	vocab.update(tokens)

# 加载文件夹中所有文件
def process_docs(directory, vocab):
	# 遍历文件夹下所有文件
	for filename in listdir(directory):
		# 跳过cv9开头的测试数据集
		if filename.startswith('cv9'):
			continue
		# 构建完整的文件路径
		path = directory + '/' + filename
		# 加载文件，生成词列表，计数
		add_doc_to_vocab(path, vocab)
        
# 将词列表保存到词汇文件
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()



# 定义词列表
vocab = Counter()
# add all docs to vocab
process_docs(r'E:\临时测试程序\pytest\txt_sentoken\pos', vocab)
process_docs(r'E:\临时测试程序\pytest\txt_sentoken\neg', vocab)
# 打印词列表长度
print(len(vocab))
# 打印词列表中前50个单词单词
print(vocab.most_common(50))

# 保留列表中出现两次以上的词
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))


# 将词列表保存到词汇文件
save_list(tokens, r'E:\临时测试程序\pytest\vocab.txt')