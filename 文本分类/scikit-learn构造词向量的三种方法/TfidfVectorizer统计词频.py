from sklearn.feature_extraction.text import TfidfVectorizer
# 下面是一个文本文档的列表
text = ["The quick brown fox jumped over the lazy dog.",
        "The dog.",
        "The fox"]
# 实例化过程
vectorizer = TfidfVectorizer()
# 标记并建立索引
vectorizer.fit(text)
# 输出 单词-索引 字典
print(vectorizer.vocabulary_)
# 查看词频评分
print(vectorizer.idf_)
# 编码文档,将评分归一化
vector = vectorizer.transform([text[0]])
# 查看编码后的向量
print(vector.shape)
print(vector.toarray())