from gensim.models.wrappers import FastText
import time

begin=time.clock()
print('Load model')
model = FastText.load_fasttext_format('/opt/fasttext/model/skipgram.bin')
end=time.clock()
print('Lode model done,time: {}'.format(end-begin))


print(model.most_similar(positive='电视机'))
# Output = [('headteacher', 0.8075869083404541), ('schoolteacher', 0.7955552339553833), ('teachers', 0.733420729637146), ('teaches', 0.6839243173599243), ('meacher', 0.6825737357139587), ('teach', 0.6285147070884705), ('taught', 0.6244685649871826), ('teaching', 0.6199781894683838), ('schoolmaster', 0.6037642955780029), ('lessons', 0.5812176465988159)]
print(model['科沃斯'])


print(model.similarity('科沃斯', '机器人'))

# Output = 0.683924396754