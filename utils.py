#!/usr/bin/env python
# coding: utf-8

"""
    Utils
    ============================
    For model and text classificatinn

    _copyright_ = 'Copyright (c) 2018 J.W. - Everis', 
    _license_ = GNU General Public License
"""

import pandas as pd
import numpy as np
import re
import string
import pickle

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
#!pip install stop_words
from stop_words import get_stop_words

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import preprocessing

import es_core_news_sm  #es_core_news_md
from spacy import displacy
from spacy.tokenizer import Tokenizer

import matplotlib.pyplot as plt
import seaborn as sns

from flashtext import KeywordProcessor


### Parámetros generales
resource_path = 'release3/resources'

# Variables no utilizadas
rem_var = ['Column1','Column2',...,'ColumnX']

# Some key phrases of class 1
predefined_phrases1 = ['phrases1','phrases2',..,'phrasesX']

# Some key phrases of class 2
predefined_phrases2 = ['phrases1','phrases2',..,'phrasesX']

### Parámetros del modelo,
CV = 5 # K fold value


### Funciones para la carga de archivos
def read_excel(path, sheet_name='Hoja1'):
    df = pd.read_excel(path=path, sheet_name=sheet_name)
    print('Data size: ', df.shape)
    return df

### Funciones de preprocesamiento
def remove_variables(df, rem_var=rem_var):
    """Remove variables that are not used"""
    df['NUMDAYS'] = (df['FECHAFINATENCION'] - df['FECHAINGRESO']).dt.days
    df = df.drop(rem_var, axis=1)
    df = df.dropna()
    return df

def remove_rows(df, target, category, show=True):
    df = df[df[target] !=category]
    if show:    print(df.groupby(target).TEXTCOLUMNNAME.count())
    return df

def target_handling(df, target, cat_to_id='category_to_id.pkl', id_to_cat='id_to_category.pkl'):
    """ Text target categories to id dictionaries"""
    df[f'{target}_id'] = df[f'{target}'].factorize()[0]
    category_id_df = df[[f'{target}', f'{target}_id']].drop_duplicates().sort_values(f'{target}_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[[f'{target}_id', f'{target}']].values)

    print("Saving target categories")
    with open(f'{resource_path}/{cat_to_id}', 'wb') as f:
        pickle.dump(category_to_id, f)
    with open(f'{resource_path}/{id_to_cat}', 'wb') as f:
        pickle.dump(id_to_category, f)

    return category_id_df, id_to_category, df

def onehot_encode(data, onehot='onehot_encoder.pkl'):
    """One hot encoding using sklearn 
    data: numpy array"""
    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(data)
    
    with open(f'{resource_path}/{onehot}', 'wb') as f:
        pickle.dump(onehot_encoder, f)

    return onehot_encoded


### Normalización de texto
def remove_numbers(text):
    return ''.join([letter for letter in text if not letter.isdigit()])
 
def remove_punctuation(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub(' ', str(text))

def clean_text(text):
    text = remove_punctuation(text)
    text = remove_numbers(text)
    
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

def generate_stopwords(stopname='stopSpanish.pkl'):
    """ Remove stop words, and apply stemming """
    stemmer=SpanishStemmer()
    stopwords_es = set(stopwords.words('spanish'))
    stopwords_es_sw = set(get_stop_words('spanish'))
    stopSpanishBeta = list(set(stopwords_es.union(stopwords_es_sw)))

    stopSpanish = set(stopwords_es.union(stopwords_es_sw))
    for stopWord in stopSpanishBeta:
        stopSpanish.add(stemmer.stem(stopWord))

    stopSpanish = list(stopSpanish)
    stopSpanish.extend(['tra', 'd', 'desc']) # Adding stopwords not present in the standard stopwords
    stopSpanish.remove('no')  # Keep to help identify negative categories

    with open(f'{resource_path}/{stopname}', 'wb') as f:
        pickle.dump(stopSpanish, f)

    return stopSpanish

def remove_stopwords(text, stopSpanish):
    stemmer=SpanishStemmer()
    textList = text.split()
    textList = [word for word in textList if word not in stopSpanish]
    return ' '.join([stemmer.stem(word) for word in textList])

### Funciones para extraer features mas relevantes
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids.  None means across all documents'''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


### Funciones para extraer features adicionales
def keyword_gen(category, stopSpanish):
    if category == 'cat1':
        phrases = predefined_phrases1
    elif category == 'cat2':
        phrases = predefined_phrases2
    else:
        phrases = phrases

    keyword_processor = KeywordProcessor()
    for phrase in phrases:
        clean_phrase = remove_stopwords(clean_text(phrase), stopSpanish)
        keyword_processor.add_keyword(clean_phrase)
    return keyword_processor

def look_features(kp, text):
    kf = kp.extract_keywords(text)
    if len(kf)>0:
        return 1
    else: return 0
    

### Funciones para visualizar
def plot_freq_target_comentario(df, target):
    fig = plt.figure(figsize=(8,6))
    df.groupby(target).TEXTCOLUMNNAME.count().plot.bar(ylim=0)
    plt.show()

    print(df.groupby(target).TEXTCOLUMNNAME.count())


def plot_confusion_matrix(y_test, y_pred, category_id_df):
    "Confusion matrix"
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=category_id_df.TARGET.values, yticklabels=category_id_df.TARGET.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_metrics(y_test, y_pred, target):
    "PLot precision, recall and f1-score"
    print(metrics.classification_report(y_test, y_pred, 
                                        target_names=df[target].unique()))


def plot_tfidf_classfeats_h(dfs, id_to_category):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(14, 35), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(5, 2, i+1)  # len(dfs)/2
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(id_to_category[df.label]), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

### Funciones para guardar resultados finales o intermedios
def confusion_matrix_csv(y_test, y_pred, matrix_name='cmatrix.csv'):
    d = {'actual':y_test, 'predicted':y_pred}
    tdf = pd.DataFrame(d)
    tdf.to_csv(f'{resource_path}/{matrix_name}', index=False)
