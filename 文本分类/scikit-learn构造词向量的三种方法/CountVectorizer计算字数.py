from sklearn.feature_extraction.text import CountVectorizer
"""
原理：
1.先对文本词汇进行索引编码，构造出 “单词-索引” 字典
2.文本索引构建原理为：按照单词字母顺序排序，a排在前，z排在后
3.将每个文本编辑成一个一维向量
3.1向量的长度为词汇表中的总单词数
3.2向量中元素排序为索引顺序从小到大排序
3.3向量中元素数字是单词出现的次数
eg:
单词-索引 字典： dic：{a:1,b:2,c:3,d:4,e:5,:f:6,g:7}
转换前的文本：text = bcfadde
转换后的向量：vec = [1,1,1,2,1,1,0]  (2代表d出现2次，0代表g没出现过，1代表出现过一次的单词)
https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
"""
# 待转换文本
text = ["The quick brown fox jumped over the lazy dog."]
# 创建一个基于CountVectorizer的，词向量构建法对象
vectorizer = CountVectorizer()
# 使用CountVectorizer构造出 “单词-索引” 字典
vectorizer.fit(text)
# 输出构造出的字典
print(vectorizer.vocabulary_)
# 使用 “单词-索引” 字典，将换文本数据转换成数字序列
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(type(vector))
# 输出转换后向量结果
print(vector.toarray())


text2 = ["the puppy dog"]
vector = vectorizer.transform(text2)
print(vector.toarray())