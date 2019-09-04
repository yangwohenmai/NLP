from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
"""
本文构建了一个 Embedding层-过渡层-分类层 的简单3层网络，
通过训练网络，使得Embedding层学习出对应的权重矩阵，作为“嵌入层参数”，可供以后直接使用
数据构造方法:
1.定义词汇量
2.调用one_hot()，根据词汇量对文本进行onehot编码
3.调用pad_sequences(),将编码后的向量补全成相同长度
4.构建网络训练Embedding层
5.训练得到Embedding层的权重矩阵就是嵌入层参数
https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
"""
# 10个文本
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
# 文本对应结果
labels = array([1,1,1,1,1,0,0,0,0,0])
# 定义词汇量
vocab_size = 50
# 对10个文本分别进行one hot编码
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
# 每个文本的补全长度为4
max_length = 4
# 将one hot编码好的 不同长度的向量 文档补全成相同长度
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

model = Sequential()
# 定义嵌入层，参数：词汇量，每个单词8维度，每个文本长度为4
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# 拟合模型
model.fit(padded_docs, labels, epochs=160, verbose=2)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))