#!/usr/bin/python

##############################################################
# module: find_usenet_posts.py
# Cole Shepherd
# a01895001
# bugs to vladimir dot kulyukin at usu dot edu
###############################################################

import os
import sys
import sklearn.datasets
import scipy as sp
from sklearn.cluster import KMeans
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer

## Get the USENET POSTS
usenet_posts = sklearn.datasets.fetch_20newsgroups()

## Sample user posts
user_post1 = \
    """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""

user_post2 = 'is fuel injector cleaning necessary?'
user_post3 = 'are diesel engines more fuel efficient than gas ones?'
user_post4 = 'how many european players play in the nhl?'

## create objects for the Snowball and Porter stemmers.
snowball_stemmer = nltk.stem.SnowballStemmer('english')
porter_stemmer = nltk.stem.PorterStemmer()

class SnowballTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(SnowballTfidfVectorizer, self).build_analyzer()
        return lambda doc: (snowball_stemmer.stem(w) for w in analyzer(doc))

class PorterTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(PorterTfidfVectorizer, self).build_analyzer()
        return lambda doc: (porter_stemmer.stem(w) for w in analyzer(doc))
    
## let's create two vectorizer objects.
SnowballVectorizer = SnowballTfidfVectorizer(min_df=10,
                                    stop_words='english',
                                    decode_error='ignore')
PorterVectorizer = PorterTfidfVectorizer(min_df=10,
                                        stop_words='english',
                                        decode_error='ignore')

def compute_feat_mat(data, vectorizer):
    feat_mat = vectorizer.fit_transform(data)
    return feat_mat

def fit_km(feat_mat, num_clusters):
    km = KMeans(n_clusters=num_clusters, n_init=1, verbose=1, random_state=3)
    clustered_data = km.fit(feat_mat)
    return clustered_data

def find_posts_similar_to(post, feat_mat, data, vectorizer, km, top_n=10):

    new_post_vec = vectorizer.transform([post]).getrow(0).toarray()
    km_predicted_labels = km.predict(new_post_vec)
    print('len(km_predicted_labels)=%d' % len(km_predicted_labels))

    top_new_post_label = km.predict(new_post_vec)[0]

    posts_in_same_cluster = (km.labels_ == top_new_post_label).nonzero()[0]

    similar_posts = []

    for i in posts_in_same_cluster:
        dist = sp.linalg.norm(new_post_vec - feat_mat[i])
        similar_posts.append((dist, data[i]))


    similar_posts.sort(key=lambda post: post[0])

    list = []

    for x in xrange(top_n):
        list.append(similar_posts[x])

    return list




