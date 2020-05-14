import numpy as np
from time import sleep
from prep import prepTREC
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from readtrec import getAbstract

qrels = []
for line in open('qrels-rnd1.txt').readlines():
	vals = line.strip().split(" ")
	#topic, cord_uid, qrel, assessround
	qrels.append([int(vals[0]), vals[3], float(vals[4]), float(vals[1])])

qrels = np.array(qrels, dtype="O")

manual_classification = [2,0,3,0,3,7,3,7,6,5,4,5,0,0,0,0,7,5,5,1,0,0,1,1,1,7,7,3,3,3]
#manual_classification_rnd2 = [2,0,3,0,3,6,3,6,7,5,4,5,0,0,0,0,3,5,5,1,0,0,1,1,6,6,6,3,3,3,2,2,3,1,9]

#store documents for each task here
doc_sets = []      #contains qrels per task
abstract_sets = [] #contains individual fulltext abstracts per task
combined_abstracts = [] #contains all abstracts in a text appended into one big doc
for i in range(10):
	doc_sets.append([])
	abstract_sets.append([])

#we add one more here - for abstracts that are not in a task
abstract_sets.append([])

#for each topic, get all documents 
for topic in range(1, 31):
	docs = qrels[qrels[:,0] == topic]
	
	#add these documents to the documents for each task
	for np_array in docs:
		doc_sets[manual_classification[topic - 1]].append(list(np_array))

# we have no examples of 8 and 9 - which makes sense when you look at those task descriptions
print(doc_sets[1])


#built by looking at results
sw = stopwords.words("english")
#sw.extend(['the', 'of', 'abstract', 'background', 'nan', 'in', 'is', 'to', 'coronavirus', 'covid', 'novel', 'outbreak', 'disease', 'and', 'an', 'we', 'has', 'this', 'are', 'summary', 'december'])
porter = PorterStemmer()

# now get the abstracts for these items
def getAbstract(cord_id, metadata):
	abstract = metadata[metadata['cord_uid'] == cord_id]['abstract']

	#do some preprocessing
	abstract = abstract.to_string().lower()
	#remove tags (probably unnecessary)
	abstract = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", abstract)
	#remove special chars and digits
	abstract = re.sub("(\\d|\\W)+", " ", abstract)

	#Whoopsie.. we don't want stemmed query expansion terms
#	words = abstract.split(" ")
#	words = [porter.stem(word) for word in words]
#	abstract = " ".join(words)
	
#	(porter.stem(abstract.split(" ")))
	return abstract

metadata = prepTREC('./docids-rnd1.txt')
for task_id, task_docs in enumerate(doc_sets):
	for doc in task_docs:
#		print(doc)
		abstract_sets[task_id].append(getAbstract(doc[1], metadata))
	combined_abstracts.append(" ".join(abstract_sets[task_id]))
	
#finally, we also want to use the other abstracts for our corpus (so our tf idf scores get better)
corpus = ""
for row in np.array(metadata):
	if row[0] not in qrels[:,0]:
		#print(row[8])
		#corpus += str(row[8])
		abstract_sets[10].append(str(row[8]))
	
print()
print('Now start computing tf-idf')
cv = CountVectorizer(max_df=0.85, min_df=0.05, stop_words = sw)

#all_docs = combined_abstracts[:].append(corpus)
word_count_vector = cv.fit_transform(abstract_sets[10])

tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
print(tfidf_transformer)

feature_names = cv.get_feature_names()

#helper functions for below
def sort_coo(coo_matrix):
	tuples = zip(coo_matrix.col, coo_matrix.data)
	return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
	
def get_topn(feature_names, sorted_vectors, topn=10):
	sorted_vectors = sorted_vectors[:topn]
	
	scores = []
	features = []
	
	for ind, score in sorted_vectors:
		scores.append(round(score, 3))
		features.append(feature_names[ind])
		
	results = {}
	for ind in range(len(features)):
		results[features[ind]] = scores[ind]
		
	return results

task_keywords = []
for task_id, task_text in enumerate(combined_abstracts):
	tfidf_vector = tfidf_transformer.transform(cv.transform([task_text]))

	#sort tf idf vectors by scores
	sorted_vectors = sort_coo(tfidf_vector.tocoo())
	
	#get top keywords for task w/ score
	keywords = get_topn(feature_names, sorted_vectors, 50)
	
	print(keywords)
	task_keywords.append(keywords)


#We need to filter words that occur in every description

#For each task
new_stopwords = []
for taskid, keywords in enumerate(task_keywords):
	#for all words
	for word in keywords:
		counter = 0
		#check how often this keyword occurs in other tasks
		for t in task_keywords:
			if word in t:
				counter += 1 
		if counter > 3:
			new_stopwords.append(word)

print(new_stopwords)

for sw in new_stopwords:
	for task in task_keywords:
		task.pop(sw, None)#del task[sw]

print(task_keywords)