import logging

from gensim import corpora, models, similarities
from gensim.models import TfidfModel
from gensim import similarities

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# taken from https://radimrehurek.com/gensim/wiki.html#wiki
dictionary = corpora.Dictionary.load_from_text('corpus/wiki_pap_wordids.txt.bz2')
corpus_tfidf = corpora.MmCorpus('corpus/wiki_pap_tfidf.mm')

lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  # initialize an LSI transformation

corpus_lsi = lsi_model[corpus_tfidf]

# tfidf = TfidfModel.load('corpus/wiki_pap.tfidf_model')

doc = "un tipo di mamífero protehá for di aña"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi_model[vec_bow]  # convert the query to LSI space
print(vec_lsi)
# transform corpus to LSI space and index it
index = similarities.MatrixSimilarity(lsi_model[corpus_lsi])

sims = index[vec_lsi]  # perform a similarity query against the corpus
print(sims)
