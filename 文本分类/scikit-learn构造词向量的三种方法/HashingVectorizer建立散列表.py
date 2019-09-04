from sklearn.feature_extraction.text import HashingVectorizer
"""
统计个数CountVectorizer和计算频率IF-IDF两种方法，其局限性导致词汇量可能变得非常大，词向量要占用大量内存
通过哈希散列 则不需要专门建立索引，并且可以将向量的长度设定为任意值
缺点是哈希散列是一个单向函数，无法将编码转换回单词
HashingVectorizer这个矢量化器不需要调用fit()函数来训练数据文档。实例化之后可以直接用于编码文档
样例文档将被编码为包含20个元素的稀疏数组。
编码文档的值默认将字数标准化到 -1 和 1 之间，这里也可以通过更改默认配置使其进行简单的整数计数。
https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
"""
text = ["The quick brown fox jumped over the lazy dog."]
# 设定特征值为20（词向量维度为20）
vectorizer = HashingVectorizer(n_features=20)
# 将文本中的单词进行哈希散列
vector = vectorizer.transform(text)
print(vector.shape)
# 输出散列后的词向量
print(vector.toarray())

text = ["The dog."]
vectorizer = HashingVectorizer(n_features=20)
vector = vectorizer.transform(text)
print(vector.shape)
print(vector.toarray())

text = ["The dog,dog"]
vectorizer = HashingVectorizer(n_features=20)
vector = vectorizer.transform(text)
print(vector.shape)
print(vector.toarray())
