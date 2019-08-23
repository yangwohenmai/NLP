import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

"""
在线下载的是已经被IMDB数字序列化之后的文本
训练文本句子是转化好的数字序列，结果是0,1对应的情感标志
加载前5000个最长用的单词作为词汇表
评论取前500个字，不足500补0
每个字向量取32维
根据上述参数构建词嵌入层
嵌入层->平滑层->全连接层->输出层
"""
# 只加载IMDB数据集前5000个最常用单词，作为词汇表
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words = 500
# 每条评论获取前500个字，不足500填充0
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# create the model
model = Sequential()
# 嵌入层 权重矩阵大小为500*32 每次输入500个单词 每个word为向量32维 词汇表为5000
model.add(Embedding(top_words, 32, input_length=max_words))
# 将嵌入层转换成一维输入
model.add(Flatten())
# 250节点 整流全连接层
model.add(Dense(250, activation='relu'))
# 输出层为单神经元 sigmoid函数用于分类
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 数据量比较多，所以步长可设置为128
model.fit(X_train, y_train, epochs=3, batch_size=128, verbose=2)
# validation_data=(X_test, y_test)，将测试数据集代入训练（不建议）
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128, verbose=2)
# 评估
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))