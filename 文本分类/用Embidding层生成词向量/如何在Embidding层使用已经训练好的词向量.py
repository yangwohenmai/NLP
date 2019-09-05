from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
# 定义文本词汇量
vocab_size = len(t.word_index) + 1
# 用整数索引编码文本
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# 每个文本向量长度为4
max_length = 4
# 对文本进行补齐
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

embeddings_index = dict()
# 加载glove.6B.100d文件
f = open('E:\\临时测试程序\\pytest\\glove.6B.100d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# 给文本创建一个（行:词汇量，列:100维）的0矩阵，用于存放后续文本
embedding_matrix = zeros((vocab_size, 100))
# 用glove.6B.100d中的词向量，与文本中单词进行匹配
# 此处相当于使用训练好的词向量glove来对文本docs进行编码，而不在网络中重新训练词向量
# 最后生成的矩阵embedding_matrix就是文本docs通过词向量glove转换成的，词向量矩阵
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
# define model
model = Sequential()
# weights=[embedding_matrix]表示将文本转换好的词向量矩阵直接代入嵌入层使用，而不在网络中重新训练嵌入层
# trainable=False表名网络训练过程中不对嵌入层的词向量矩阵进行修改
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

model.fit(padded_docs, labels, epochs=50, verbose=0)

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))