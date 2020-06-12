import numpy as np
from time import sleep
from prep import prepTREC
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from readtrec import getAbstract

qrels = []
for line in open('qrels-rnd2.txt').readlines():
	vals = line.strip().split(" ")
	#topic, cord_uid, qrel, assessround
	qrels.append([int(vals[0]), vals[3], float(vals[4]), float(vals[1])])

qrels = np.array(qrels, dtype="O")

manual_classification = [2,0,3,0,3,6,3,6,7,5,4,5,0,0,0,0,3,5,5,1,0,0,1,1,6,6,6,3,3,3,]
#manual_classification = [2,0,3,0,3,6,3,6,7,5,4,5,0,0,0,0,3,5,5,1,0,0,1,1,6,6,6,3,3,3,2,2,3,1,9]

#store documents for each task here
doc_sets = []      #contains qrels per task
journal_sets = []  #contains individual fulltext abstracts
for i in range(10):
	doc_sets.append([])
	journal_sets.append([])

#for each topic, get all documents 
for topic in range(1, 35):
	docs = qrels[qrels[:,0] == topic]
	
	#add these documents to the documents for each task
	for np_array in docs:
		doc_sets[manual_classification[topic - 1]].append(list(np_array))

# we have no examples of 8 and 9 - which makes sense when you look at those task descriptions
#print(doc_sets[1])

# now get the abstracts for these items
def getJournal(cord_id, metadata):
	journal = metadata[metadata['cord_uid'] == cord_id]['journal']

	#do some preprocessing
	journal = journal.to_string()
	#print(" ".join(journal.split(" ")[4:]))
	
	return " ".join(journal.split(" ")[4:])

metadata = prepTREC('./docids-rnd3.txt')

for task_id, task_docs in enumerate(doc_sets):
	for doc in task_docs:
#		print(doc)
		journal_sets[task_id].append(getJournal(doc[1], metadata))
	


import pandas as pd
from collections import Counter

print(np.array(journal_sets).shape)
for task_id, task_journals in enumerate(journal_sets):
	print(task_journals)
	if(len(task_journals) > 0):
		pd.Series(task_journals).value_counts().plot('bar')
		plt.show()
#	letter_counts = Counter(task_journals)
#	print(letter_counts)
#	df = pandas.DataFrame.from_dict(letter_counts, orient='index')
#	df.plot(kind='bar')
