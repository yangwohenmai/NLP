# NLP网络
NLP自然语言处理学习代码汇总  
## 想学习更多深度学习项目，可访问如下链接
1.通过自回归(AR,ARM)模型进行时间序列预测合集：https://github.com/yangwohenmai/TimeSeriesForecasting  
2.通过深度学习模型进行时间序列预测合集：https://github.com/yangwohenmai/DeepLearningForTSF  
3.基于LSTM的时间序列预测项目合集：https://github.com/yangwohenmai/LSTM  

## 一、文本分类
### 1.IMDB_LSTM+X 4种文本分类  
```
  1.简单LSTM分类  
  1.简单LSTM分类_通用版  
  2.LSTM层间dropout  
  3.LSTM层内dropout  
  4.LSTM+卷积网络  
  4.LSTM+卷积网络_通用版  
  vocab2.txt
```
+ ***txt_sentoken***  
```
  neg  
  pos  
```  
### 2.IMDB两个简单的文本分类示例 
```
  1.IMDB在线数据使用方法  
  2.简单的多层感知器分类模型(在线数据)  
  3.一维卷积神经网络模型(在线数据)  
```  
### 3.Keras将Word嵌入层(Embidding)用于深度学习  
```
  1.如何训练Embidding层  
  2.在Embidding层使用已训练好的词向量_glove  
```
### 4.scikit-learn构造词向量的三种方法  
```
  1.CountVectorizer计算字数  
  2.HashingVectorizer建立散列表  
  3.TfidfVectorizer统计词频(TF-IDF)  
```
### 5.第三方中文分词器  
```
  1.三种分词器  
```
### 6.实例_基于word2vec词向量分类  
```
  1.清理数据  
  2.词汇计数器  
  3.准备正面和负面评论  
  4.训练Embedding模型  
  5.调用Embedding模型分类  
  6.训练可用于嵌入层的word2vec词向量  
  7.将训练好的word2vec转化为嵌入层  
  embedding_word2vec.txt  
  vocab.txt  
  vocab2.txt  
```
+ ***txt_sentoken***  
```
  neg  
  pos  
```  
### 7.实例_基于词袋模型分类  
```
  1.清理数据  
  2.词汇计数器  
  3.准备正面和负面评论  
  4.生成词向量模型  
  5.构建NLP网络  
  6.调用词袋模型开始预测  
  vocab.txt   
```
### 8.实例_文本分类的多通道CNN模型  
```
  1.清洗评论并保存文本  
  2.训练、保存模型  
  3.三通道CNN网络分类  
  4.五通道CNN网络分类  
  test.pkl  
  train.pkl  
```
### 9.使用Gensim生成词嵌入  
```
  1.开发Word2Vec嵌入  
  2.使用PCA绘制单词向量  
  3.词向量的减法  
```
### 10.网络参考代码  
```
  IMDB文本分类  
  IMDB文本分类_改进  
  TextCNN  
```
## 二、文本生成
### 1.基于字符的神经语言模型
```
  1.语言模型设计  
  2.训练语言模型  
  3.使用网络生成文本  
  char_sequences.txt  
  mapping.pkl  
  model.h5  
  rhyme.txt  
```
### 2.
```
  01.
  02.
  03.
```
### 3.
```
  01.
  02.
  03.
```