import numpy
from keras.datasets import imdb
from matplotlib import pyplot

# 加载所有在线文本数据
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)

# 查看加载的文本数据集的数据结构，各50000条
print("Training data: ")
print(X.shape)
print(y.shape)

# 训练集y数据类型，[0,1]类型
print("Classes: ")
print(numpy.unique(y))

# 文本数据拼接
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X))))

# 计算评论总词汇数量
print("Review length: ")
result = [len(x) for x in X]
# 每个文本平均长度234，标准差为172
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))
# plot review length
pyplot.boxplot(result)
pyplot.show()
