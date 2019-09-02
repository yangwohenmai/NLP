from sklearn.feature_extraction.text import CountVectorizer
"""
原理：
1.先对文本词汇进行索引编码，构造 单词-索引 字典
2.文本索引构建原理为：按照单词字母顺序排序，a排在前，z排在后
3.将每个文本编辑成一个一维向量
向量的长度为词汇表中的总单词数
向量中元素排序为索引顺序从小到大排序
向量中元素数字是单词出现的次数
dic：{a:1,b:2,c:3,d:4,e:5,:f:6,g:7}
text = bcfadde
vec = [1,1,1,2,1,1,0]  2代表d出现2次，0代表g没出现过，1代表出现过一次的单词
"""
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
# encode another document
text2 = ["the puppy dog"]
vector = vectorizer.transform(text2)
print(vector.toarray())