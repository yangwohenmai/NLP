# LSTM for sequence classification in the IMDB dataset
from string import punctuation
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from os import listdir
from numpy import array
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# 特殊符号处理
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0  # 没有出现但字典里有的词
word_index["<START>"] = 1# 起始符号
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# 获取“单词-数字”对应字典
change_word_index = dict([(key, value) for (key, value) in word_index.items()])

# 将文本转化为数字
def text_to_index(text, change_word_index):
    try:
        ll = list()
        ll.append(1)
        ll.extend([change_word_index.get(word,0) for word in text])
        #llarray = numpy.asarray(ll, dtype = int)
        #string = ' '.join('%s' %id for id in ll)
    except Exception as e:
        print(e)
    return ll


# 加载文件
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r',encoding="utf-8")
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
	#tokens = ' '.join(tokens)
	return tokens

# 文本清洗
def process_docs(directory, is_trian):
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
        # 
        tokens_index = text_to_index(tokens, change_word_index)
        # 将处理后的评论文本加入列表
        documents.append(tokens_index)
    return numpy.asarray(documents)

# 加载生成好的词汇列表vocab
vocab_filename = r'D:\MyGit\NLP\文本分类\LSTM_IMDB文本分类\vocab2.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


xtrain_pos = process_docs(r'D:\MyGit\NLP\文本分类\基于word2vec词向量分类\txt_sentoken/pos', True)
xtrain_neg = process_docs(r'D:\MyGit\NLP\文本分类\基于word2vec词向量分类\txt_sentoken/neg', True)
xtrain = xtrain_neg.extend(xtrain_pos)
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# truncate and pad input sequences
max_review_length = 500
#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
xtrain = sequence.pad_sequences(xtrain, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#model.fit(X_train, y_train, epochs=3, batch_size=64)
model.fit(xtrain, ytrain, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))