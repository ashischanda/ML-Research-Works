#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:00:48 2017

@author: ashis
"""
# USE python 3.6
# install gensim
# collect data from: http://mattmahoney.net/dc/text8.zip
# Tutorials: http://nbviewer.jupyter.org/github/danielfrg/word2vec/blob/master/examples/word2vec.ipynb
# http://nbviewer.jupyter.org/github/danielfrg/word2vec/blob/master/examples/doc2vec.ipynb

#https://rare-technologies.com/word2vec-tutorial/
#https://www.kaggle.com/c/word2vec-nlp-tutorial#part-3-more-fun-with-word-vectors
#https://www.kaggle.com/c/word2vec-nlp-tutorial#description


import word2vec

'''
# we already created that all. So, simply load on memory
word2vec.word2phrase('text8', 'text8-phrases', verbose=True)

#Train the model using the word2phrase output.
word2vec.word2vec('text8-phrases', 'text8.bin', size=100, verbose=True)

# creating cluster. (Running K mean to the vector space)
word2vec.word2clusters('text8', 'text8-clusters.txt', 100, verbose=True)
                     
'''

model = word2vec.load('text8.bin') #So, simply load on memory

#print (model.vocab)         # print vocabulary list
#print (model.vocab.shape)   # Total word numbers (98331,)
#print (model.vectors)       # vector values

print (model['dog'].shape)   # each word has 100 features/columns
print (model['dog'][:10])



indexes, metrics = model.cosine('teacher')  # finding top similar vectors and word's index
#model.vocab[indexes]    # printing words using the index
print ( model.generate_response(indexes, metrics).tolist() ) # printing word and vector value together
'''
[('doctor', 0.8172569307240375), 
('student', 0.7650909362962448), 
('pupil', 0.7594323056183803), 
('physician', 0.7367708114107208), 
('tutor', 0.7329538126240454), 
('professor', 0.7276049326667439), 
('master', 0.7274431367309091), 
('friend', 0.7241185553350069), 
('mentor', 0.7051743380884687), 
('teaching', 0.7019425679383207)]
'''
      
      
      
      
# Since we trained the model with the output of word2phrase we can ask for similarity of "phrases"
      
indexes, metrics = model.cosine('los_angeles')
print (model.generate_response(indexes, metrics).tolist() )  # san_francisco  , san_diego .......

print ()
indexes, metrics = model.cosine('salary')
print (model.generate_response(indexes, metrics).tolist() ) # budget, earnings, spending, pay

print ()
indexes, metrics = model.cosine('address')
print (model.generate_response(indexes, metrics).tolist() ) # server, client, update


print ()
indexes, metrics = model.cosine('age')
print (model.generate_response(indexes, metrics).tolist() ) # birth, years, died

print ()
indexes, metrics = model.cosine('highest')
print (model.generate_response(indexes, metrics).tolist() ) # lowest, ranking, higher


# ********************************************
# Analogies
# Its possible to do more complex queries like analogies such as: king - man + woman = queen 
# This method returns the same as cosine the indexes of the words in the vocab and the metric

indexes, metrics = model.analogy(pos=['king', 'woman'], neg=['man'], n=10)

# **************************************************************************



# ****************************** now look at the cluster
# ********* In cluster file, you can find each word has a number. it presents cluster number
clusters = word2vec.load_clusters('text8-clusters.txt')
print (clusters.get_words_on_cluster(90).shape )        # 255.  It means 255 words in this cluster
print (clusters.get_words_on_cluster(90) )              # It prints all words of the cluster

print (  clusters[b'the'] )                             # 49. 'The' is belong to 49 cluster      
print (clusters.get_words_on_cluster(49) )
print (clusters.get_words_on_cluster(49)[:10] )         # showing first 10 words
  



# We can add the clusters to the word2vec model and generate a response that includes the clusters

model.clusters = clusters
indexes, metrics = model.analogy(pos=['paris', 'germany'], neg=['france'], n=10)
print (model.generate_response(indexes, metrics).tolist() )
'''
[('berlin', 0.3148203622259862, 20), 
('munich', 0.28484810833477464, 5), 
('vienna', 0.2827875147408495, 82), 
('leipzig', 0.27928420570213297, 41), 
('moscow', 0.27573777005200484, 59), 
('st_petersburg', 0.2564100469092737, 63), 
('dresden', 0.25221583705941997, 86), 
('prague', 0.24953067507179635, 19), 
('hamburg', 0.24538174874487068, 98), 
('z_rich', 0.24536478649384624, 42)]
'''

