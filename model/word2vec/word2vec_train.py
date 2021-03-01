'''
CBOW와 Skip-gram
'''

import sys
sys.path.append('../..')

from model.common.trainer import Trainer
from model.common.optimizer import Adam
from model.word2vec.simple_cbow import SimpleCBOW
from model.common.util import preprocess, create_contexts_target, convert_one_hot


window_size = 1
hidden_size = 5
batch_size = 3

max_epoch = 1000

text = 'Let\'s move to anotehr word'
corpus, word_to_idx, idx_to_word = preprocess(text)

vocab_size = len(word_to_idx)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
# trainer.plot()


word_vecs = model.word_vecs
for word_id, word in idx_to_word.items():
    print(word, word_vecs[word_id])