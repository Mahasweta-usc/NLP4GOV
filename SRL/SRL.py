
import os, sys
import torch
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import random
np.random.seed(0)
random.seed(0)

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk import word_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('universal_tagset')

all_words = []
tags = ['SCONJ','CCONJ','DET','INTJ','NUM']
for elem in brown.tagged_words(tagset="universal"):
  if elem[1] in tags:
    if elem[0].lower() not in ['no','not']: all_words.append(elem[0].lower()) ##all determiners except negation

all_words = list(set(all_words))

import sklearn
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import CountVectorizer
stopwords = list(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
stopwords.extend(all_words)

import ast
import itertools
from tqdm import tqdm
tqdm.pandas()
import json

import re
import stanza
stanza.download('en')

import spacy
nlp_spacy = spacy.load("en_core_web_sm")

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse',use_gpu=True)
import string
import pprint


# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import math
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.stats import chi2_contingency
from scipy.special import eval_chebyc

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
plt.rcParams["figure.figsize"] = (11.7,8.27)
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

main_arguments = ['ARG' + str(idx) for idx in range(6)]

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import matplotlib.pyplot as plt

"""##**Extract Predicate Arguments**##"""

#Download AllenAI model
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",cuda_device=torch.cuda.current_device())
#SRL arguments of interest
arg_types = ['ARGM-GOL','ARGM-COM','ARGM-NEG','ARGM-MOD','ARGM-DIR','ARGM-LOC','ARGM-MNR','ARGM-TMP','ARGM-ADV','ARGM-PRP','ARGM-ADJ','ARGM-LVB','ARGM-CAU' ,
             'ARGM-PNC','ARGM-EXT','ARGM-REC','ARGM-PRD','ARGM-DIS','ARGM-DSP','ARGM-RLC','ARG0','ARG1','ARG2','ARG3','ARG4','ARG5','ARG6','V']
predictor._model = predictor._model.cuda()

"""##**Class & Methods**##"""

class SRL:
  def __init__(self, agent="strict"):
        self.agent = agent

  #pass list of sentences as [{'sentence':...},{'sentence':...}]
  def srl_arg(self, sentences):
    parsed = predictor.predict_batch_json(
        sentences
    )
    outcome = dict()
    for sentence,parse in zip(sentences,parsed):
      sentence = sentence['sentence']
      verb_list = []
      for elem in parse['verbs']:
        parsed_items = dict()
        for item in arg_types:
          arg_found = re.findall("\[{}: (.*?)]".format(item), elem['description'])
          if len(arg_found) : parsed_items[item] = arg_found
        if bool(parsed_items) and 'V' in parsed_items: verb_list.append(parsed_items)
      outcome[sentence] = verb_list
    return outcome

#############################################################################################################################################
                                          ### SIDDIKI'S METHODS START FROM BELOW ###
#############################################################################################################################################

  #read a file and apply preprocessing
  def file_read(self,file_name):

    if isinstance(file_name,str): data = pd.read_csv(file_name)
    else: data = file_name

    data.columns = map(str.lower, data.columns)
    data.dropna(subset=['raw institutional statement'],inplace=True)
    ##currently not considering multi level coding
    data.drop_duplicates(subset=['raw institutional statement'],inplace=True)
    data.fillna('', inplace=True)
    data = data[data['aim']!=""]
    data = data.applymap(lambda x : str(x).lower().strip())

    # data['sentences'] = data['raw institutional statement'].apply(lambda x : [sentence.text.lower() for sentence in nlp(x).sentences])
    # data = data.explode('sentences')
    data['sentences'] = data['raw institutional statement'].apply(lambda x : x.lower())

    #find root verb through stanza
    data['ROOT'] = data['sentences'].apply(lambda x : [word.text for sent in nlp(x).sentences for word in sent.words  if word.deprel == 'root'][0])
    data = data[(data['ROOT'] != '')]

    data['srl_ip'] = data['raw institutional statement'].apply(lambda x : [{'sentence' : x}])
    data['srl_parsed'] = data.apply(lambda x: self.srl_arg(x['srl_ip'])[x['raw institutional statement']],axis=1)
    data = data[data['srl_parsed'].map(lambda d: len(d)) > 0]

    data = data.explode('srl_parsed')
    data['srl_verb'] = data['srl_parsed'].apply(lambda x : x['V'][0])
    #only keep frame parsed for root verbs and has agents/objects
    data['keep'] = data.apply(lambda x : (x['ROOT'] == x['srl_verb']) & (any(elem in x['srl_parsed'] for elem in main_arguments)),axis=1)

    return data[data['keep']]

  def detect_sub(self,text):
      doc = nlp_spacy(text)
      sub_toks = [tok for tok in doc if 'subj' in tok.dep_ ]
      if sub_toks: return True
      else: return False
  #argument matching
  def argmatch(self, x, text, arg):
    if arg == 'attribute_inf':
        if 'ARG0' in x:
            return ", ".join(x['ARG0'])

        if self.detect_sub(text) and 'ARG1' in x:
            return ", ".join(x['ARG1'])

        return ""



    if arg == 'object_inf':
        keys = sorted(list(set(main_arguments) & set(x.keys())))
        if 'ARG0' in x:
            keys.remove('ARG0')
            if keys: return ", ".join(x[keys[0]])

        if self.detect_sub(text) and 'ARG1' in x:
            keys.remove('ARG1')

        if keys : return ", ".join(x[keys[0]])
        else: return ""

    if arg == 'aim_inf': return " ".join(x['V'])
    if arg == 'deontic_inf':
        if 'ARGM-MOD' in list(x.keys()) : return " ".join(x['ARGM-MOD'])
        else: return ""

  # normalize the texts
  def normalize_text(self,s):
      """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

      def remove_stopwords(text):
          # regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
          # return re.sub(regex, " ", text)
          return [word for word in text if word not in all_words]

      def tokens(text):
          doc= nlp(text)
          return ([word.lemma for sent in doc.sentences for word in sent.words if not word.text in all_words])
          # return word_tokenize(text)

      def remove_punc(text):
          exclude = set(string.punctuation)
          return "".join(ch for ch in text if ch not in exclude)

      def lower(text):
          return text.lower().strip()

      return tokens(remove_punc(lower(s)))

  def compute_exact_match(self,prediction, truth):
      return int(self.normalize_text(prediction) == self.normalize_text(truth))

  #F1 score computation
  def compute_f1(self,truth, prediction):
    pred_tokens = self.normalize_text(prediction)
    truth_tokens = self.normalize_text(truth)
    # print(pred_tokens,truth_tokens)

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

  def remove_outliers(self, data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    cleaned_data = [x for x in data if lower_bound <= x <= upper_bound]
    return cleaned_data

  def inference(self,file_name,out_path=None):
    # if 'Food' in file_name: self.agent= None
    # else: self.agent = 'strict'
    # if not out_path : out_path = file_name
    data = self.file_read((file_name))


    data.columns = map(str.lower, data.columns)
    ##exclude empty entries
    data.dropna(subset=['raw institutional statement'],inplace=True)
    data = data[(data['raw institutional statement'] != "")]

    for arg in ['attribute_inf','object_inf','aim_inf','deontic_inf']:
      data[arg] = data.apply(lambda x : self.argmatch(x.srl_parsed,x.sentences,arg),axis=1)



    data.to_csv(os.path.join(out_path),index=False)

  def srl_eval(self):
    column_names = ['attribute', 'object', 'deontic', 'aim']
    eval_scores = {name:[] for name in column_names}

    datasets = os.listdir('/content/IG-SRL/SRL/data')
    # file_names = ['NationalOrganicProgramRegulations_Siddiki.xlsx - Econ Development Mechanisms.csv']

    for subdata in datasets:
        sub_path = os.path.join('/content/IG-SRL/SRL/data', subdata)
        sets = []
        for file_name in os.listdir(sub_path):
            eval_name = os.path.join('/content/IG-SRL/SRL/data', subdata,file_name)
            temp = pd.read_csv(eval_name)
            temp.columns = map(str.lower, temp.columns)
            sets.append(temp[['raw institutional statement','attribute','deontic','aim','object']])


        df1 = pd.concat(sets)
        print(subdata)
        out_path = os.path.join('/content',f"{subdata}_data_new.csv")
        self.inference(df1, out_path)
        df1 = pd.read_csv(out_path)
        df1 = df1[['raw institutional statement','attribute','deontic','aim','object','attribute_inf','object_inf','aim_inf','deontic_inf']]


        ##exclude statements with no aim coded
        df1.replace("",np.nan,inplace=True)
        df1.dropna(subset=column_names,inplace=True,how='all')
        df1.dropna(subset=['aim'],inplace=True)

        df1.fillna("", inplace=True)
        df1 = df1.applymap(lambda x : x.lower().strip())


        for col_name in column_names:
            df = df1.copy()
            ##remove inferred coding re.sub("[\(\[].*?[\)\]]", "<skipped>",x)
            pattern = r'\[[^\]]*\]'
            df[col_name] = df[col_name].apply(lambda x : "<skipped>" if x.startswith('[') else x)
            df[col_name] = df[col_name].apply(lambda x : re.sub("[\(\[].*?[\)\]]", "",x))

            ##remove inferred coding
            # df.replace("<skipped>", "", inplace=True)
            df = df[df[col_name] != '<skipped>']
            values1 = df[col_name].tolist()
            values2 = df[col_name + '_inf'].tolist()

            #do we need to compare missing entries?
            values1 = [str(x).replace('nan', '') for x in values1]
            values2 = [str(x).replace('nan', '') for x in values2]

            f1_score = []
            for x,y in zip(values1,values2):
                f1_score.append((self.compute_f1(x,y)))
                eval_scores[col_name].append(self.compute_f1(x,y))

            print(f" F1 score for {col_name}: {np.mean(f1_score)} for {df1.shape[0]} entries")
        df1.to_csv(out_path,index=False)

    # Create a 2x2 subplot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    for idx, col_name in enumerate(column_names):
        # eval_scores[col_name] = self.remove_outliers(eval_scores[col_name])

        # Plot each boxplot in a different subplot
        row, col = int(idx/2),int(idx%2)
        axes[row, col].boxplot(eval_scores[col_name])
        axes[row, col].set_title(f"{col_name}")

    fig.suptitle("Span selection match (F1) across multiple datasets")
    return plt

