import os
#import jieba
import pkuseg
#from pyltp import Segmentor

lexicon = ['经', '少安', '贺凤英', 'F-35战斗机', '埃达尔·阿勒坎'] # 自定义词典

# 哈工大LTP分词
def ltp_segment(sent):
    # 加载文件
    cws_model_path = os.path.join('data/cws.model') # 分词模型路径，模型名称为`cws.model`
    lexicon_path = os.path.join('data/lexicon.txt') # 参数lexicon是自定义词典的文件路径
    segmentor = Segmentor()
    segmentor.load_with_lexicon(cws_model_path, lexicon_path)
    words = list(segmentor.segment(sent))
    segmentor.release()

    return words

# 结巴分词
def jieba_cut(sent):
    for word in lexicon:
        jieba.add_word(word)
    return list(jieba.cut(sent))

# pkuseg分词
def pkuseg_cut(sent):
    seg = pkuseg.pkuseg(user_dict=lexicon)
    words = seg.cut(sent)
    return words

sent = '尽管玉亭成家以后，他老婆贺凤英那些年把少安妈欺负上一回又一回，怕老婆的玉亭连一声也不敢吭，但少安他妈不计较他。'
#sent = '据此前报道，以色列于去年5月成为世界上第一个在实战中使用F-35战斗机的国家。'
#sent = '小船4月8日经长江前往小鸟岛。'
#sent = '1958年，埃达尔·阿勒坎出生在土耳其首都安卡拉，但他的求学生涯多在美国度过。'

#print('ltp:', ltp_segment(sent))
#print('jieba:', jieba_cut(sent))
print('pkuseg:', pkuseg_cut(sent))
