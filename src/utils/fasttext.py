import gensim
from gensim.models.wrappers import FastText
import time

begin=time.clock()
print('Load model')
model = FastText.load_fasttext_format('/media/ecovacs/DATA/model/fasttext/cbow.bin')
# model = gensim.models.Word2Vec.load('/opt/word2vec/benebot_vector/word2vec.bin')
end=time.clock()
print('Lode model done,time: {}'.format(end-begin))

l=['彩电','电视','冰箱','手机','电吹风','西门子','Dyson','科斯','ecovacs','步步高','新华书店','吴中','小明','辛健','松下','猪八戒']
for word in l:
    if model.__contains__(word.strip()):
        print(word,model.most_similar(positive=word))
    else:
        print('OOV',word)
# Output = [('headteacher', 0.8075869083404541), ('schoolteacher', 0.7955552339553833), ('teachers', 0.733420729637146), ('teaches', 0.6839243173599243), ('meacher', 0.6825737357139587), ('teach', 0.6285147070884705), ('taught', 0.6244685649871826), ('teaching', 0.6199781894683838), ('schoolmaster', 0.6037642955780029), ('lessons', 0.5812176465988159)]
# print(model['科沃斯'])
# print(model[''])
# print(model[' '])
#
#
# print(model.similarity('AI', '机器人'))

# Output = 0.683924396754