#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:57:17 2018

@author: timothylucas
"""

import numpy as np
import newspaper
from goose3 import Goose
import os
from mitie import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster

# get newsarticles and save them in format described in the article
# Filename: title-<article_numer>.txt, example: title-1.txt
# Contents: The title of the news story.
# Filename: article-<article_numer>.txt, example: article-1.txt
# Contents: The contents of the news story.
# And optionally, also store the topic of the news article
# Filename: topic-<article_numer>.txt, example: topic-1.txt
# Contents: The actual “topic” (section or sub-section) under which the news story was
#           classified on the hosting website.

def get_articles(path, news_website = 'https://www.yahoo.com/news/', max_articles = 150):
    # articles should be saved in /articles file
    # See Goose and newspaper3k documentation for explanation on how to use
    # these packages (tried to use Beautiful Soup for this but was 
    # frustratingly difficult, since it seems that the newspages load when
    # someone is actually on the side, instead of automatically loading 
    # everything).
    # https://github.com/goose3/goose3 for goose documentation
    os.chdir(path)
    paper = newspaper.build(news_website)
    g = Goose()
    i = 0
    for article in paper.articles:
        if 'html' in article.url:
            i += 1
            print(article.url)
            print(i)
            if i != max_articles:
                url = article.url
                article_extr = g.extract(url=url)
                file = open('title-{}.txt'.format(i),'w') 
                file.write(article_extr.title)  
                file.close()
                file = open('article-{}.txt'.format(i),'w') 
                file.write(article_extr.cleaned_text)
                file.close()
                file = open('topic-{}.txt'.format(i),'w') 
                file.write(article_extr.domain)
                file.close()
            else:
                break
        else:
            continue

if __name__ == '__main__':
    
    # First, get the articles from the function
    curr_path = os.getcwd()
    articles_path = curr_path+'/articles'
    
    # Use line below if you didn't get the articles
    #get_articles(articles_path, news_website = 'https://www.yahoo.com/news/', max_articles = 200)
    
    # total number of articles to process
    N = 150
    # in memory stores for the topics, titles and contents of the news stories
    topics_array = []
    titles_array = []
    corpus = []
    for i in range(1, N+1):
        # get the contents of the article
        with open('article-{}.txt'.format(i), 'r') as myfile:
            d1=myfile.read().replace('\n', '')
            d1 = d1.lower()
            corpus.append(d1)
        #get the original topic of the article
        with open('topic-{}.txt'.format(i), 'r') as myfile:
            to1=myfile.read().replace('\n', '')
            to1 = to1.lower()
            topics_array.append(to1)
        #get the title of the article
        with open('title-{}.txt'.format(i), 'r') as myfile:
            ti1=myfile.read().replace('\n', '')
            ti1 = ti1.lower()
            titles_array.append(ti1)
        
    # NER path
    
    path_to_ner_model = '/Users/timothylucas/Documents/Personal/MITxPRO/Libs/MITIE-models/english/ner_model.dat'
    ner = named_entity_extractor(path_to_ner_model)
    
    # entity subset array
    
    entity_text_array = []
    for i in range(1, N+1):
        # Load the article contents text file and convert it into a list of words.
        tokens = tokenize(load_entire_file(('article-{}.txt'.format(i))))
        # extract all entities known to the ner model mentioned in this article
        entities = ner.extract_entities(tokens)
        # extract the actual entity words and append to the array
        for e in entities:
            range_array = e[0]
            tag = e[1]
            score = e[2]
            score_text = '{:0.3f}'.format(score)
            entity_text = b' '.join([tokens[j] for j in range_array])
            entity_text_array.append(entity_text.lower())
    
    # remove duplicate entities detected
    entity_text_array = np.unique(entity_text_array)

    # Construct TfidVectorizer
    vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',
                           stop_words='english', vocabulary=entity_text_array)
    corpus_tf_idf = vect.fit_transform(corpus)
    
    # change n_clusters to equal the number of clusters desired
    n_clusters = 7
    n_components = n_clusters
    #spectral clustering
    spectral = cluster.SpectralClustering(n_clusters= n_clusters,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors",
                                          n_neighbors = 17)
    spectral.fit(corpus_tf_idf)
    
    if hasattr(spectral, 'labels_'):
        cluster_assignments = spectral.labels_.astype(np.int)
    for i in range(0, 40): #len(cluster_assignments))
        # removed topic cluster here because the site I used (yahoo)
        # didn't have very good topics by default
        print('Document number : {}'.format(i))
        print('Cluster Assignment : {}'.format(cluster_assignments[i]))
        print('Document title : {}'.format(titles_array[i]))
        print('------------------------')