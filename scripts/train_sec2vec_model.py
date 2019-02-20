import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import collections
import smart_open
import random
import logging
import os
import sys
import multiprocessing

# Set file names for train and test data
# test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
# lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
# lee_test_file = test_data_dir + os.sep + 'lee.cor'
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield TaggedDocument(gensim.utils.simple_preprocess(line), [i])

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]
    train_file = inp
    train_corpus = list(read_corpus(train_file))
    model = Doc2Vec(vector_size=100, min_count=1, window=2, workers=multiprocessing.cpu_count())
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(outp1)
    # model.wv.save(outp2, binary=False)
# lee_train_file = '../data/raw/SemEval2010_train_raw.txt'
# lee_test_file = '../data/SemEval2010/train/C-41.txt.final'
#
#
#
# train_corpus = list(read_corpus(lee_train_file))
# test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
# model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
# model.build_vocab(train_corpus)
# model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
# model.save(outp1)
# model.wv.save_word2vec_format(outp2, binary=False)
# print(model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires']))
