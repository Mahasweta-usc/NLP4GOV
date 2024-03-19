import os
from os import posix_spawn
import torch
os.system('python -m spacy download en_core_web_sm')

import stanza
stanza.download('en')

import numpy as np
import re
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)

# import allennlp_models.tagging
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",cuda_device=torch.cuda.current_device())


PRN = ["her", "herself", "him", "himself", "his", "hisself", "it", "itself", "me", "myself", "one", "oneself", "ours",
"ourselves", "ownself", "self",  "theirs", "them", "themselves", "they",'their', "she", "he", "us", 'you', 'yours', "this", "that", "these", "those"]
poss_prn = ["her","his","ours","theirs",'their','yours']

class corefs:
  def __init__(self):
    self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,ner,depparse',use_gpu=True)
  def prns(self, test_list,indices):
    for i in sorted(indices, reverse=True):
        del test_list[i]
        test_list.insert(i, " ")
    return test_list

  def coref(self, text):
    text = str(text)
    text = text.lower()
    text = text.replace('\n'," ")
    text = re.sub(' +', ' ', text)

    corefs = predictor.predict(document=text)
    clusters= corefs['clusters']
    document = corefs['document']

    span_rep, spans = [], []


    # iterate through clusters
    for cluster in clusters:
      identity = None

      for elem in cluster:
        #chose most descriptive span as replacement
        spanids = document[elem[0]:elem[1]+1]
        if not all(item in PRN for item in spanids) :
          temp = " ".join(spanids)
          if len(temp) > len(identity): identity = temp

      if not identity: continue

      for id, elem in enumerate(cluster): elem.append(identity); spans.append(elem)

    #remove nested spans in clusters

    for elem in spans:
      check = set(range(elem[0],elem[1]+1))
      if not any(check < set(range(elem2[0],elem2[1]+1)) for elem2 in spans) : span_rep.append(elem)


    span_rep = sorted(span_rep, key= lambda x:x[1], reverse = True)

    for pair in span_rep:
      start = pair[0]
      span = range(pair[0],pair[1]+1)
      identity = pair[2]
      rem = " ".join(document[pair[0]:pair[1]+1])
      document = self.prns(document,span)

      if rem not in poss_prn : document[start] = identity
      else: document[start] = identity + "'s"

    text = " ".join(document)

    return re.sub(' +', ' ', text)