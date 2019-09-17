from gensim.models import Word2Vec
"""
size :(默认为100）嵌入的维数，例如表示每个标记（字）的密集向量的长度。
window :(默认值5）目标字与目标字周围的字之间的最大距离。
min_count :(默认值5）训练模型时要考虑的最小字数; 出现小于此计数的单词将被忽略。
workers :(默认值3）训练时使用的线程数。
sg :(默认为0或CBOW）训练算法，CBOW（0）或skip-gram（1）。
"""
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# 直接用Word2Vec模型来训练文本，生成嵌入层，min_count过滤词频小于1的词
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# 输出Word2Vec模型的形状
words = list(model.wv.vocab)
print(words)
# 输出sentence单词的词向量
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)