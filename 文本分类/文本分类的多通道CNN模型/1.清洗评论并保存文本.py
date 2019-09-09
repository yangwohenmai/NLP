from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from pickle import dump

# 加载原始评论文本
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# 文本内容处理
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# 删除标点符号
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# 删除非字母字符
	tokens = [word for word in tokens if word.isalpha()]
	# 过滤掉停用词
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# 只取长度大于2的单词
	tokens = [word for word in tokens if len(word) > 1]
	tokens = ' '.join(tokens)
	return tokens

# 对文本进行数据清洗
def process_docs(directory, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		path = directory + '/' + filename
		# 加载文本
		doc = load_doc(path)
		# 清理文本
		tokens = clean_doc(doc)
		# 将清理后的文本加入list
		documents.append(tokens)
	return documents

# 保存清洗后的评论文本
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)

# 加载训练数据
negative_docs = process_docs(r'E:\临时测试程序\pytest\txt_sentoken/neg', True)
positive_docs = process_docs(r'E:\临时测试程序\pytest\txt_sentoken/pos', True)
trainX = negative_docs + positive_docs
# 前900条为负面评论，后900条为正面评论
trainy = [0 for _ in range(900)] + [1 for _ in range(900)]
save_dataset([trainX,trainy], 'E:\MyGit\TEST\pytest/train.pkl')

# 加载测试数据
negative_docs = process_docs(r'E:\临时测试程序\pytest\txt_sentoken/neg', False)
positive_docs = process_docs(r'E:\临时测试程序\pytest\txt_sentoken/pos', False)
testX = negative_docs + positive_docs
# 前100条为负面评论，后100条为正面评论
testY = [0 for _ in range(100)] + [1 for _ in range(100)]
#保存清洗后的“评论文本-评论情感”数据
save_dataset([testX,testY], 'E:\MyGit\TEST\pytest/test.pkl')