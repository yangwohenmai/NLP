from sklearn.feature_extraction.text import TfidfVectorizer
"""
传说中的IF-IDF分词法，通过词频，计算词汇得分，构造词向量
词频（Term Frequency）：表示给定单词在这份文档中出现的频率。
逆向文件频率（Inverse Document Frequency）：用于降低其他文档中普遍出现的单词的最终评分。
TF-IDF可以突出个性化的单词，例如在只在这份文档中频繁出现，但其他文档中较少出现的单词
"""
# 下面是一个文本文档的列表
text = ["The quick brown fox jumped over the lazy dog.",
        "The dog.",
        "The fox"]
# 创建一个基于TF-IDF的，词向量构建法对象
vectorizer = TfidfVectorizer()
# 使用TfidfVectorizer构造出 “单词-索引” 字典
vectorizer.fit(text)
# 输出构造出的 单词-索引 字典
print(vectorizer.vocabulary_)
# 查看词频评分
print(vectorizer.idf_)

# 用词频评分对象 来编码文本文档，将词频评分归一化，但不是标准归一化
vector = vectorizer.transform([text[0]])
print(vector.shape)
print(type(vector))
# 查看文本被编码后的向量
print(vector.toarray())

# 用词频评分对象 来编码文本文档，将词频评分归一化，但不是标准归一化
vector = vectorizer.transform([text[1]])
print(vector.shape)
print(type(vector))
# 查看文本被编码后的向量
print(vector.toarray())

# 用词频评分对象 来编码文本文档，将词频评分归一化，但不是标准归一化
vector = vectorizer.transform([text[2]])
print(vector.shape)
print(type(vector))
# 查看文本被编码后的向量
print(vector.toarray())