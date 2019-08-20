from string import punctuation
from os import listdir
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
"""
生成的work2vec文件是给每个单词生成一个n维向量，如：
film[11,22,33,44,55....,99]
good[33,44,55,66,77....,99]
将每个单词都映射成一个n维向量
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

# 文本数据清洗
def doc_to_clean_lines(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    lines = list()
	# walk through all files in the folder
    for filename in listdir(directory):
		# skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
		# create the full path of the file to open
        path = directory + '/' + filename
		# load and clean the doc
        doc = load_doc(path)
        doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
        lines += doc_lines
    return lines

# 加载词汇文件
vocab_filename = r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# 加载训练数据
positive_docs = process_docs(r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\txt_sentoken/pos', vocab, True)
negative_docs = process_docs(r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\txt_sentoken/neg', vocab, True)
sentences = negative_docs + positive_docs
print('训练数据总数量为: %d' % len(sentences))

# 训练 word2vec 模型,size=100表示每个词对应的向量为100维
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)

# 统计模型中的词向量数量
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# 保存训练的词向量文件
filename = r'E:\MyGit\NLP\文本分类\基于word2vec词向量分类\embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)


## 对训练的model做出散点图
# 获取训练好的所有单词的向量（此处只对前30个点处理）
X = model[model.wv.vocab][0:30]
# 定义二维坐标
pca = PCA(n_components=2)
# 对单词向量进行PCA转义
result = pca.fit_transform(X)
# 创建投影的散点图
pyplot.scatter(result[:, 0], result[:, 1])
# 获取词向量列表（此处只对前30个点处理）
words = list(model.wv.vocab)[0:30]
# 给词向量散点图上的点，匹配对应的文字
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()