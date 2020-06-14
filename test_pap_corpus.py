import logging
import gensim
from gensim import corpora, models, similarities
from pprint import pprint
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# taken from https://radimrehurek.com/gensim/wiki.html#wiki
id2word = corpora.Dictionary.load_from_text('corpus/wiki_pap_wordids.txt.bz2')
mm = corpora.MmCorpus('corpus/wiki_pap_bow.mm')

# extract 100 LDA topics, using 1 pass and updating once every 1 chunk (10,000 documents)
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=0, passes=2)
# print the most contributing words for 20 randomly selected topics
lda.print_topics(20)

# Get an article and its topic distribution
with open("corpus/wiki_pap_bow.mm.metadata.cpickle", 'rb') as meta_file:
    docno2metadata = pickle.load(meta_file)

doc_num = 13
print("Title: {}".format(docno2metadata[doc_num][1]))

vec = mm[doc_num] # get tf-idf vector
print(lda.get_document_topics(vec))
