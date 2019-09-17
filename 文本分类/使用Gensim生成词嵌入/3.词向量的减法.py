from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = r'E:\临时测试程序\pytest\glove.6B.100d.txt'
word2vec_output_file = r'E:\临时测试程序\pytest\glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)


# load the Stanford GloVe model
filename = r'E:\临时测试程序\pytest\glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['sex', 'game'], negative=['player'], topn=3)
print(result)

