'''
将文本形式的影评分为“正面”或“负面”影评。这是一个二元分类（又称为两类分类）的示例，也是一种重要且广泛适用的机器学习问题。

TensorFlow 中包含 IMDB 数据集。我们已对该数据集进行了预处理，
将影评（字词序列）转换为整数序列，其中每个整数表示字典中的一个特定字词。
'''
import tensorflow as tf 
from tensorflow import keras
from keras.datasets import imdb
import numpy as np
#from plot import plot


# 参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词。为确保数据规模处于可管理的水平，罕见字词将被舍弃。
# imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 看一下训练集的大小
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# 看一下训练集啥样 这里是影评 一个数字对应字典中的一个单词
print(train_data[0],'\n',len(train_data[0]))
print(train_data[1],'\n',len(train_data[1]))
print(type(train_data))
# 可以看到文本的长度并不相同 但神经网络一般需要相同维数的向量进行输入 下面可以看到解决办法



# 将整数转换回字词
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0  # 没有出现但字典里有的词
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text]) # 从字典中取出对应的单词  没有就取'?'

# test 将整数转换回字词
print(decode_review(train_data[0]))



# 为了使输入的张量维数相同 我们取max_length作为维数
# 我们将使用 pad_sequences 函数将长度标准化
train_data = keras.preprocessing.sequence.pad_sequences(train_data, 
                                                        value = word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, 
                                                       value = word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)                            

#  查看处理之后的数据
print(train_data[0],'\n',len(train_data[0]))


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
# 该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。
# 模型在接受训练时会学习这些向量。这些向量会向输出数组添加一个维度。
model.add(keras.layers.Embedding(vocab_size,16)) # batch * sequence * 16
# 通过对序列维度求平均值，针对每个样本返回一个长度固定的输出向量。
# 这样，模型便能够以尽可能简单的方式处理各种长度的输入。
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary() # 看一下模型的框架


# 模型在训练时需要一个损失函数和一个优化器。由于这是一个二元分类问题
# 且模型会输出一个概率（应用 S 型激活函数的单个单元层），
# 因此我们将使用 binary_crossentropy 损失函数。
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 从原始训练数据中分离出 10000 个样本，创建一个验证集(validation)。
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 用有 512 个样本的小批次训练模型 40 个周期。这将对 x_train 和 y_train 张量中的所有样本进行
#  40 次迭代。在训练期间，监控模型在验证集的 10000 个样本上的损失和准确率：
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=22,
                    batch_size=512,
                    validation_data=(x_val, y_val), # 实时验证
                    verbose=1)

# 评估模型
results = model.evaluate(test_data, test_labels)
print(results)

# model.fit() 返回一个 History 对象，该对象包含一个字典，其中包括训练期间发生的所有情况：
history_dict = history.history
print(history_dict.keys())

# 调用另一个文件里的函数 进行作图
# plot(history_dict)
