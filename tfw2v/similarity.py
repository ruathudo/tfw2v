import pickle
import itertools
import pandas as pd

from multiprocessing import cpu_count
import parmap

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from gensim.similarities import SparseMatrixSimilarity
from gensim.summarization.textcleaner import split_sentences


class TFW2V:
    def __init__(self):

        self.se = None
        self.vecs = None
        self.dictionary = None
        self.result = None


    def run(self, text, wv, stopwords=None, min_tfidf=0.1, lim_token=20, alpha=0.1, lim_most=1):
        # preprocess the text
        se, tokens = self._preprocessing(text, stopwords)
        # train tfidf model
        dictionary, vecs = self._train_tfidf(tokens)

        self.se = se
        self.dictionary = dictionary
        self.vecs = vecs

        self.result = self._enrich_tfidf(wv, min_tfidf=min_tfidf, lim_token=lim_token, alpha=alpha, lim_most=lim_most)

        return self.result


    def _preprocessing(self, text, stopwords):
        if type(text) is list:
            se = pd.Series(text)
        elif type(text) is pd.Series:
            se = pd.Series(text.to_list())    #copy as new series
        else:
            raise Exception("The text must be in List type or pandas Series type!")


        tokens = se.apply(simple_preprocess, max_len=50)
        # remove stopwords
        if stopwords:
            tokens = tokens.apply(lambda x: [w for w in x if w not in stopwords])

        return se, tokens


    def _train_tfidf(self, tokens):
        # tokens is series data type of tokenized text of each document
        # doc_tokens = df['tokens'].apply(lambda x: list(itertools.chain.from_iterable(x)))

        # generate dict and corpus
        dictionary = Dictionary(tokens)
        corpus = tokens.apply(dictionary.doc2bow).tolist()
        # train tf-idf
        tfidf = TfidfModel(dictionary=dictionary)
        # calculate sparse matrix vecs for corpus
        vecs = tfidf[corpus]

        # dictionary.save(os.path.join(model_dir, 'dictionary.dict'))
        # print('trained tf-idf')

        return dictionary, vecs


    def train_w2v(self, text, size=100, epochs=20):

        sents = text.apply(split_sentences)

        flatten_sents = pd.Series(itertools.chain.from_iterable(sents.tolist()))
        token_sents = flatten_sents.apply(simple_preprocess, max_len=50)

        #model = FastText(size=100, window=5, min_count=1, word_ngrams=1, sg=1, negative=5, workers=cpu_count())
        model = Word2Vec(size=size, window=5, min_count=1, sg=1, negative=5, workers=cpu_count())

        model.build_vocab(sentences=token_sents)
        model.train(sentences=token_sents, total_examples=len(token_sents), epochs=epochs)

        # if save_path:
        #     model.save(os.path.join(save_path))

        # print('trained word2vec')

        return model


    def _recalculate(self, doc_id, vecs, sim_index, dictionary, wv, min_tfidf=0.1, lim_token=20, alpha=0.1):
        vec = vecs[doc_id]
        tokens = sorted(vec, key=lambda x: x[1], reverse=True) # (token_id, tf_score)
        filterred_tokens = list(filter(lambda x: x[1] <= min_tfidf, tokens))
        tokens = filterred_tokens if len(filterred_tokens) else tokens[:lim_token]
        
        sim_docs = sim_index[vec]
        
#         if doc_id < 3:
#             print(sim_docs)
        
        new_sim_docs = []
        
        compared_words = [dictionary[x[0]] for x in tokens]
        #print(compared_words, '\n')
        
        for i, s in sim_docs:   # doc_id, cosine similarity score
            sim_vec = vecs[i]
            #sim_limit =  int(frac * len(sim_vec))
            sim_tokens = sorted(sim_vec, key=lambda x: x[1], reverse=True) # (token_id, tf_score)
            filterred_sim_tokens = list(filter(lambda x: x[1] <= min_tfidf, sim_tokens))
            sim_tokens = filterred_sim_tokens if len(filterred_sim_tokens) else sim_tokens[:lim_token]
            
            sim_words = [dictionary[x[0]] for x in sim_tokens]
            
            #print(sim_words, '\n')
            
            sim_score = wv.n_similarity(compared_words, sim_words)
            
            new_sim_docs.append((i, (sim_score * alpha + s) / (1 + alpha)))

#             if doc_id < 3:
#                 print(compared_words, '\n')
#                 print(sim_words, '\n')
#                 print(sim_score, '\n')
#                 print('current sim score', s, '\n')
#                 print('new sim score', new_sim_docs[i], '\n')

        new_sim_docs = sorted(new_sim_docs, key=lambda x: x[1], reverse=True)

        return new_sim_docs


    def _enrich_tfidf(self, wv, min_tfidf=0.1, lim_token=20, alpha=0.1, lim_most=1):
        vecs = self.vecs
        dictionary = self.dictionary
        corpus_size = len(vecs)
        
        sim_index = SparseMatrixSimilarity(vecs, num_features=len(dictionary), num_best=int(corpus_size*lim_most))

        # print(sim_index)
        result = {}

        # for each doc, get the list of similar docs in sorted order
        # for top k percent of similar docs, recalculate the similarity scores
        # for each top n words in this doc, compare with top m words in other docs
        # calculate the bonus point by combining tfidf score and similarity score from w2v model
        # save bonus point in a list, add up later to the sim matrix
        doc_ids = range(corpus_size)
        result = parmap.map(self._recalculate, doc_ids, vecs, sim_index, dictionary, wv, min_tfidf=min_tfidf, lim_token=lim_token, alpha=alpha)

        return result


    def save(self, path):
        with open(path + '_vecs.pkl', 'wb') as fp:
            pickle.dump(self.vecs, fp)

        with open(path + '_result.pkl', 'wb') as fp:
            pickle.dump(self.result, fp)

        self.dictionary.save(path + '_dictionary.pkl')
        self.se.to_pickle(path + '_text.pkl')


    def load(self, path):
        self.dictionary = Dictionary.load(path + '_dictionary.pkl')
        self.se = pd.read_pickle(path + '_text.pkl')

        with open(path + '_vecs.pkl', 'rb') as fp:
            self.vecs = pickle.load(fp)

        with open(path + '_result.pkl', 'rb') as fp:
            self.result = pickle.load(fp)
        

    def most_similar(self, idx, k=10):
        # map text with the similar docs given a doc id
        
        indices = [x[0] for x in self.result[idx]]
        # convert text to dataframe
        sub = self.se.iloc[indices][:k]
        return sub
