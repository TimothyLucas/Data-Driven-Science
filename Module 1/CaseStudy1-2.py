#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:43:03 2018

@author: timothylucas
"""

## First, write the webscraper for ArXiv

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as n
import nltk
from nltk.corpus import stopwords
import collections

#####
## GETTING DATA
#####

# First find the names of all of the faculty members

def findFacultyMembers(faculty_url = 'https://www.eecs.mit.edu/people/faculty-advisors'):
    faculty_page = requests.get(faculty_url)
    faculty_page_content = BeautifulSoup(faculty_page.content, 'html.parser')
    names_cont = faculty_page_content.select('div.views-field-title span.card-title a')
    names = [n.get_text() for n in names_cont]
#    departments_cont =  faculty_page_content.select('div.views-field-term-node-tid span.field-content a')
#    departments = [n.get_text() for n in departments_cont]
    return names

# Then query all of these names in arXiV, and scrape the extracts
# For now store in a DF (papername, authors, abstract)
    
def scrapeArXiV(names):
    ## TO DO:
    # 1. Guy called Arvind has a space before his name, how to remove this
    # 2. Some regexes aren't working properly, that's why there is an ugly
    #    if in some of the list comprehensions, how to fix this?
    # 3. ...
    # Names are 'firstname lastname' format
    # arxiv.org/find/(subject)/1/au:+(lastname)_(initial)/0/1/0/all/0/1
    column_names = ['title', 'unique_id', 'authors', 'abstract']
    all_papers = pd.DataFrame(columns = column_names)
    
    # for the unique id we need to find only the arXiV one
    regexp = re.compile(r'arXiv')

    
    for name in names:
        print('Now searching for {}'.format(name))
        # In the search below quotation marks have been put around the author to
        # make sure we get an EXACT match
        search_url = 'https://arxiv.org/search/?query="{}"&searchtype=author&order=&size=100'.format(name.replace(' ', '+'))
        papers_author = requests.get(search_url)
        papers_author_content = BeautifulSoup(papers_author.content, 'html.parser')
        
        # First check if there are actually any results, otherwise skip to the
        # next name  (beautiful soup checks if the page gives a warning text)
        if not papers_author_content.select('p.has-text-warning'):
            titles = papers_author_content.select('p.title')
            titles_text = [t.get_text() for t in titles]
  
            unique_id = papers_author_content.select('p.list-title a')
            unique_id_text = [u.get_text() for u in unique_id if regexp.search(u.get_text())]
            
            authors = papers_author_content.select('p.authors')
            authors_text = [[auths.get_text() for auths in a.select('a')] for a in authors]
            
            abstracts = papers_author_content.select('span.abstract-full')
            abstracts_text = [a.get_text() for a in abstracts]
          
            # Now save the found abstracts to a DataFrame
            
            name_results = pd.DataFrame({'title' : titles_text, \
                                         'unique_id' : unique_id_text, \
                                         'authors' : authors_text, \
                                         'abstract' : abstracts_text})
            
            # And append it to the dataframe that needs to be returned 
    
            all_papers = pd.concat([all_papers, name_results])
        else:
            continue
        
    return all_papers
        
# Abstracts are nicely in a dataframe now, so lets preprocess

####
# PREPROCESS DATA
####    

# Tokenize and make a word count on each abstract, probably best if we just put
# this in the dataframe

def word_cleaning_and_count(s):
     cleaning_set = set(stopwords.words('english'))
     cleaning_set.update(['△', '.', '(', ')'])
     s_lower = s.lower()
     tokens = nltk.word_tokenize(s_lower)
     word_dict = dict(collections.Counter(tokens))
     for key in cleaning_set:
         word_dict.pop(key, None)
     return word_dict

# Next function is only to make sure that the inputs are useful for the 
# SVILDA algorithm

def word_cleaning_only(s):
    re_filter = re.compile(r'\\|/|\d|~|_|\*')
    cleaned_word_list = []
    cleaning_set = set(stopwords.words('english'))
    cleaning_set.update(['△', '.', '(', ')', '$', '/'])
    s_lower = s.lower()
    tokens = nltk.word_tokenize(s_lower)
    for word in tokens:
        if (word not in cleaning_set) & (not re_filter.search(word)):
            cleaned_word_list.append(word)
    cleaned_doc = ' '.join(cleaned_word_list)
    return cleaned_doc 

####
# Actual data science
####

import stochastic_lda as slda

# First define the SVILDA as a class
# this has already been done by downloading the code from the github page
# and translating it into Py3 using lib2to3
    

if __name__ == '__main__':
    names = findFacultyMembers()
    all_papers = scrapeArXiV(names)  
    all_papers['word_dict'] = all_papers['abstract'].apply(word_cleaning_and_count)
    
    # As I don't really know what type the inputs for vocab or the docs
    # need to be, I will need to rewrite them myself. Assumed is that
    # docs : is a array that all contain all of the abstracts, I will need to clean these
    # vocab : is simply all of the words in the docs
    
    docs = list(all_papers['abstract'].apply(word_cleaning_only).values)
    vocab = dict(collections.Counter(nltk.word_tokenize(' '.join(docs))))
    
    # Set the parameters and run the function
    
    iterations = int(len(docs))
    k = 5

    testset = slda.SVILDA(vocab = vocab, K = k, D = len(docs), alpha = 0.2, eta = 0.2, tau = 1024, kappa = 0.7, docs = docs, iterations= iterations)
    testset.runSVI()
    
    finallambda = testset._lambda
    
    topics, topic_probs = testset.getTopics()
    
    for kk in range(0, len(finallambda)):
        lambdak = list(finallambda[kk, :])
        lambdak = lambdak / sum(lambdak)
        temp = list(zip(lambdak, list(range(0, len(lambdak)))))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        # print temp
        print('topic %d:' % (kk))
        # feel free to change the "53" here to whatever fits your screen nicely.
        for i in range(0, 10):
            print('%20s  \t---\t  %.4f' % (list(vocab.keys())[list(vocab.values()).index(temp[i][1])], temp[i][0]))
        print()


    with open("temp/%i_%i_%f_raw.txt" %(k, iterations, perplexity), "w+") as f:
        # f.write(finallambda)
        for result in topics:
            f.write(str(result) + " \n")
        f.write(str(topic_probs) + " \n")