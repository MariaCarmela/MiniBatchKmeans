#Import all the necessary libraries
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

import sys
from time import time

import numpy as np
import pandas as pd
import re
import string
import random

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

path='Distro'
path_corpus='Distro/corpus.txt'
path_labels='Distro/labels.txt'

#Uploading the labels file
with open(path_labels) as file_in:
    labels_orig = []
    for line in file_in:
        labels_orig.append(line)

true_k = len(np.unique(labels_orig)) ## This should be 2 in this example
print(true_k)
print('The labels are '+str(len(labels_orig)))



#Uploading the corpus file
with open(path_corpus) as file_in:
    dataset_orig = [] #each element of the list is a short text of the main corpus
    for line in file_in:
        dataset_orig.append(line)

print('The dataset has '+str(len(dataset_orig)) + ' texts')

#Sampling of the dataset:randomly choosing the index of the list and apply this randomly index to the 2 list, the dataset and the labels lists
random_indices = random.sample(range(len(dataset_orig)), len(dataset_orig)//4) # get len(dataset)//4
dataset=[]
labels=[]

for i in range(len(random_indices)):
    dataset.append(dataset_orig[random_indices[i]])    
    labels.append(labels_orig[random_indices[i]])
	
print('The labels now are '+str(len(labels)))
print('The dataset now has '+str(len(dataset)) + ' texts')


####################################################PART 1:PREPROCESSING OF THE CORPUS

#Removing Special Characters and numbers
# function to remove special characters and numbers
def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)
	
# Removing punctuation
# function to remove punctuation
def remove_punctuation(text):
    text = ''.join([c for c in text if c not in string.punctuation])
    return text


#Removing extra whitespaces and tabs
# function to remove special characters
def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()

#Tokenization and removing stopwords
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
# custom: removing words from list
stopword_list.remove('not')
# function to remove stopwords
def remove_stopwords(text):
    # convert sentence into token of words
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    # check in lowercase 
    t = [token for token in tokens if token.lower() not in stopword_list]
    text = ' '.join(t)    
    return text
	
#Applying all the preprocessing functions	
for i in range(len(dataset)):    
    dataset[i]= remove_stopwords(remove_extra_whitespace_tabs
                                         (remove_punctuation
                                          (remove_special_characters(dataset[i]))))
    
#Lemmatization
lemmatizer = WordNetLemmatizer()
for i in range(len(dataset)):
    word_list = word_tokenize(dataset[i])
    lemmatized_doc = ""    
    for word in word_list:
        lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
    dataset[i] = lemmatized_doc 



#Converting the corpus into tf-idf vectors
vectorizer = TfidfVectorizer(stop_words='english') ## Corpus is in English

X = vectorizer.fit_transform(dataset)
print(X.shape)


# ###################################################PART2:DIMENSIONALITY REDUCTION WITH TRUNCATED SVD

# Projecting the data onto the subspace(s) corresponding to the  k  main components ( k=2  in this case) using TruncatedSVD. 

print("The original data have", X.shape[1], "dimensions/features/terms")


r = true_k
t0 = time()
svd = TruncatedSVD(r)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
Y = lsa.fit_transform(X)
print("TruncatedSVD done in %fs" % (time() - t0))
var = 0
print(svd.explained_variance_ratio_)
print("The number of documents is still", Y.shape[0])
print("The number of dimension has become", Y.shape[1])

#Firt attempt of finding the most important features/terms in every topic identifying
# the words corresponding to the  10  largest components (with sign) of the right singular vector. 


terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    s = ""
    for t in sorted_terms:
        s += t[0] + " "
    print(s)
	# Create and generate a word cloud image:
    wordcloud = WordCloud( max_words=100, background_color="white").generate(s)
    wordcloud.to_file(path+"/first_attempt_t"+str(i)+".png")
# Display the generated image:
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()




#Second attempt of finding the most important features/terms in every topic by considering,
# for each singular vectors, the most important terms if we consider both ascending and descending ordering of the entries
terms = vectorizer.get_feature_names()

for i in range(svd.components_.shape[0]):
    terms_comp = [[terms[j], svd.components_[i][j]] for j in range(svd.components_.shape[1])]
    asc_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
    des_terms = sorted(terms_comp, key= lambda x:x[1], reverse=False)[:10]
    print("Topic "+str(i)+": ")
    s = ""
    for t in asc_terms:
        s += t[0] + " "
    print(s)
    # Create and generate a word cloud image:
    wordcloud = WordCloud( max_words=100, background_color="white").generate(s)
    wordcloud.to_file(path+"/second_attempt_t"+str(i)+".png")
    # Display the generated image:
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()

#############################################PART 3:CLUSTERING PROJECTED DATA WITH MINIBATCH K-MEANS

#Setting the variable we need
batch_size = 100
#centers = [[1, 1], [-1, -1]]
#n_clusters = len(centers)

#Applying MiniBatchKMeans on the projected data
mbk = MiniBatchKMeans(init='k-means++', n_clusters=true_k, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
t0 = time()
mbk.fit(Y)
t_mini_batch = time() - t0

#Evaluating the system with Homogeneity,Completeness,V-measure,Adjusted Rand-Index, Silhouette Coefficient and Accuracy
print("Clustering done in %0.3fs" % t_mini_batch)
print("Clustering done in %0.3fs" % (time() - t0))
print()
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, mbk.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, mbk.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, mbk.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, mbk.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(Y, mbk.labels_, sample_size=1000))
for i in range(0, len(labels)): 
    labels[i] = int(labels[i]) 
accuracy=metrics.accuracy_score(labels,mbk.labels_)
if accuracy< 0.500:
    print("Accuracy:%0.3f" %(1-accuracy ))
else:
    print("Accuracy:%0.3f" % accuracy)
print(mbk.cluster_centers_.shape)

#Transform the centroids of the clusters to the originale shape
original_centroids = svd.inverse_transform(mbk.cluster_centers_)

print(original_centroids.shape) ## Just a sanity check
for i in range(original_centroids.shape[0]):
    original_centroids[i] = np.array([x for x in original_centroids[i]])
svd_centroids = original_centroids.argsort()[:, ::-1]

#Finding the 10 most frequent words and plot them in a word cloud
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    text=''
    for ind in svd_centroids[i, :10]:
        text=text +' '+terms[ind]
        print(' %s' % terms[ind], end='')
        
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud( max_words=100, background_color="white").generate(text)
    wordcloud.to_file(path+"/MiniBatchKMeans_c"+str(i)+".png")
    # in order to only display the generated image without saving it in a png file uncomment the following lines
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()
    print()
    
    




#################################################PART4: APPLYING MINIBATCHK-MEANS ON THE ORIGINAL DATA (BEFORE APPLYING TRUNCATEDSVD)
#Applying MiniBatchKMeans on the original data
mbk_new = MiniBatchKMeans(init='k-means++', n_clusters=true_k, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
t0_new = time()
mbk_new.fit(X)
t_mini_batch_new = time() - t0_new

#Evaluating the system with Homogeneity,Completeness,V-measure,Adjusted Rand-Index, Silhouette Coefficient and Accuracy
print("Clustering done in %0.3fs" % t_mini_batch_new)
print("Clustering done in %0.3fs" % (time() - t0_new))
print()
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, mbk_new.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, mbk_new.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, mbk_new.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, mbk_new.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, mbk_new.labels_, sample_size=1000))
accuracy=metrics.accuracy_score(labels,mbk_new.labels_)
if accuracy< 0.500:
    print("Accuracy:%0.3f" %(1-accuracy ))
else:
    print("Accuracy:%0.3f" % accuracy)
print(mbk.cluster_centers_.shape)

#Transform the centroids of the clusters to the originale shape
original_centroids_new =mbk_new.cluster_centers_

print(original_centroids_new.shape) ## Just a sanity check
for i in range(original_centroids_new.shape[0]):
    original_centroids_new[i] = np.array([x for x in original_centroids_new[i]])
svd_centroids_new = original_centroids_new.argsort()[:, ::-1]

#Finding the 10 most frequent words and plot them in a word cloud
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    text=''
    for ind in svd_centroids_new[i, :10]:
        text=text +' '+terms[ind]
        print(' %s' % terms[ind], end='')
        
    
    # Creating and generating a word cloud image
    wordcloud = WordCloud( max_words=100, background_color="white").generate(text)
    wordcloud.to_file(path+"/MiniBatchKMeans_new_c"+str(i)+".png")
    # in order to only display the generated image without saving it in a png file uncomment the following lines
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()
    print()

