import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import re
import string
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('universal_tagset')

all_words = []
tags = ['DET','INTJ','NUM','SCONJ','CCONJ']
for elem in brown.tagged_words(tagset="universal"):
  if elem[1] in tags:
    all_words.append(elem[0])

all_words = list(set(all_words))

import sklearn
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import CountVectorizer
stopwords = list(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
stopwords.extend(all_words)

from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
word_embedding_model = SentenceTransformer("Jainam/freeflow-biencoder")
class semantic_search:

    def __init__(self):
        pass

    # function to initially screen search base using bm25
    # We lower case our text and remove stop-words from indexing

    def remove_punc(self, text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def process_(self, text):  ### Function to process text, including lowercasing, removing line breaks, duplicate words.
        text = text.replace("\r\n", " ").replace("\n", " ")
        regex = r'\b(\w+)(?:\W+\1\b)+'
        text = re.sub(regex, r'\1', text, flags=re.IGNORECASE)
        text = remove_punc(text)
        return text.lower()

    def bm25_tokenizer(self, text):  ### Custom tokenizer function for BM25. This tokenizes text and removes punctuation, stopwords, and non-alphabetic words.
        tokenized_doc = []
        for token in word_tokenize(process_(text)):

            if len(token) > 0 and token not in stopwords and token.isalpha():
                tokenized_doc.append(token)
        return tokenized_doc

    # remove elements with 0 bm25 scores

    ###Lexical search function for BM25 algorithm taking input query to search from data and give score of
    ### -best matching passage for given query from data
    def lex_search(self, query, data):
        tokenized_corpus = []
        for passage in tqdm(data["reply"]):
            tokenized_corpus.append(self.bm25_tokenizer(passage))
        bm25 = BM25Okapi(tokenized_corpus)
        ##### BM25 search (lexical search) #####
        bm25_scores = list(bm25.get_scores(self.bm25_tokenizer(query)))
        # top_n = np.argpartition(bm25_scores, -top_k)[-top_k:]
        bm25_hits = [{'corpus_id': data["reply"][idx], 'score': bm25_scores[idx]} for idx, _ in enumerate(bm25_scores)]
        filtered_bm25_hits = [entry for entry in bm25_hits if entry['score'] > 0]
        filtered_bm25_hits = pd.DataFrame(filtered_bm25_hits)
        filtered_bm25_hits.sort_values(by=['score'], inplace=True)
        return filtered_bm25_hits

    # rank matches with query by cosine distance

    ###Rank the mataches with a query by cosine distance using word embedding model with alist of data to rank and
    ###-gives out list of cosine similarity scores between the query and each item in the data.

    def dot_score(self, query, data_embed):
        ##### Re-Ranking #####
        wer = []
        query_embed = word_embedding_model.encode(query)
        for example in tqdm(data_embed):
            wer.append(util.cos_sim(query_embed, example))
        return wer
    # save csv in descending sorted order of match