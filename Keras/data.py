import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle
import pickle
import h5py
import json
import matplotlib.pyplot as plt

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
from nltk import word_tokenize

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.models import model_from_json


# importing bokeh library for interactive dataviz
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook


def ingest():

    data = pd.read_csv('/homes/gy18/Desktop/Project/data/train.csv', encoding='latin-1') # Enter your file location
    data.columns=["Sentiment","ItemID","Date","Blank","SentimentSource","SentimentText"]
    data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map( {4:1, 0:0}) #Converting 4 to 1
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', data.shape  )
    return data

def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('#'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        return tokens
    except:
        return 'NC'

def postprocess(data):
#     , n=1000000
#     data = data.head(n)
    data['tokens'] = data['SentimentText'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word] #combining w2v vectors with tfidf value of words in the tweet.
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

if __name__ == "__main__":

    data = ingest()

    tokenizer = TweetTokenizer()
    data = postprocess(data)
    LabeledSentence = gensim.models.doc2vec.LabeledSentence

    x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(1000000).tokens),
                                                        np.array(data.head(1000000).Sentiment), test_size=0.2)

    x_train = labelizeTweets(x_train, 'TRAIN')
    x_test = labelizeTweets(x_test, 'TEST')

    data_labellised = labelizeTweets(np.array(data.tokens), 'data')

    n = 1000000
    n_dim = 200
    tweet_w2v = Word2Vec(size=n_dim, min_count=10)
    tweet_w2v.build_vocab([x.words for x in tqdm(data_labellised)])

    tweet_w2v.train([x.words for x in tqdm(data_labellised)], total_examples=tweet_w2v.corpus_count,
                    epochs=tweet_w2v.iter)

    # print(tweet_w2v['bye'])

    # output_notebook()
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
                           tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                           x_axis_type=None, y_axis_type=None, min_border=1)

    word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]

    # dimensionality reduction. converting the vectors to 2d vectors
    from sklearn.manifold import TSNE

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_w2v = tsne_model.fit_transform(word_vectors)

    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
    tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]

    # plotting. the corresponding word appears when you hover on the data point.
    plot_tfidf.scatter(x='x', y='y', source=tsne_df)
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips = {"word": "@words"}
    show(plot_tfidf)

    print('building tf-idf matrix ...')
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x.words for x in data_labellised])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))

    train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
    train_vecs_w2v = scale(train_vecs_w2v)

    test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
    test_vecs_w2v = scale(test_vecs_w2v)

    print(train_vecs_w2v.shape)