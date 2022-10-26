
import nltk
import pandas as pd
import numpy as np
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
import re
import json
import requests
import plotly.graph_objects as go

text = "Japanese artist Yayoi Kusama works primarily in sculpture and installation. She first came to public attention after organizing performances in which naked participants were painted with brightly coloured polka dots—a recurring motif throughout her career."

def plot_associations(text, k=0.3, font_size=26):
   
    nouns_in_text = []
    is_noun = lambda pos: pos[:2] == 'NN'

    for sent in text.split('.')[:-1]:   
        tokenized = nltk.word_tokenize(sent)
        nouns=[word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
        nouns_in_text.append(' '.join([word for word in nouns if not (word=='' or len(word)==1)]))

    nouns_list = []
    
    for sent in nouns_in_text:
        temp = sent.split(' ')
        for word in temp:
            if word not in nouns_list:
                nouns_list.append(word)

    df_relation = pd.DataFrame(np.zeros(shape=(len(nouns_list),2)), columns=['Nouns', 'Verbs & Pres'])
    df_relation['Nouns'] = nouns_list

    is_pre_or_verb = lambda pos: pos[:2]=='JJ' or pos[:2]=='VB' or pos[:2] == 'IN'
    for sent in text.split('.'):
        for noun in nouns_list:
            if noun in sent:
                tokenized = nltk.word_tokenize(sent)
                adjectives_or_verbs = [word for (word, pos) in nltk.pos_tag(tokenized) if is_pre_or_verb(pos)]
                ind = df_relation[df_relation['Nouns']==noun].index[0]
                df_relation['Verbs & Pres'][ind]=adjectives_or_verbs
   
    return df_relation

df_relation = plot_associations(text, k=0.3, font_size=26)
print(df_relation)