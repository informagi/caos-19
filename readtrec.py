from prep import preparedoc2vec, prepTREC, get_doc_vector, list_of_tasks
import numpy as np
import xml.etree.ElementTree as ET
#in constants, add manually added class

manual_classification = [2,0,3,0,3,6,3,6,7,5,4,5,0,0,0,0,6,5,5,1,0,0,1,1,1,6,6,3,3,3]
rnd1classes = manual_classification#[2,0,3,0,3,7,3,7,6,5,4,5,0,0,0,0,7,5,5,1,0,0,1,1,1,7,7,3,3,3]
print(len(rnd1classes))

def readTopics(fname):
	root = ET.parse(fname).getroot()
	topics = []
	for num, topic in enumerate(root):
		#print(topic[0].text) #query
		topics.append([topic[0].text, rnd1classes[num]])
		#print(topic[1].text) #question
		#print(topic[2].text) #narrative
	return topics

def readAnserini(fname):
	res=[]
	f = open(fname)
	f.readline()
	for line in f.readlines():
		vals = line.strip().split(",")
		#topic, rank, cord_id, score
		res.append([vals[0], vals[1], vals[2], vals[3]])
	return res

def getAbstract(cord_id, metadata):
	#print(cord_id)
	abstract = metadata[metadata['cord_uid'] == cord_id]['abstract']
	#print(abstract.to_string())
	return abstract.to_string()

def rerank(results, model, topics, mixer=1):
	print("reranking")
	metadata = prepTREC('./docids-rnd1.txt')

	
	dists = []
	scores = []
	for result in results:
		sha = topics[int(result[0]) - 1]
#		print(sha)
		taskvector = get_doc_vector(model, sha[0])
		dist = np.linalg.norm(taskvector-get_doc_vector(model, getAbstract(result[2], metadata)))
		dists.append(dist)
		scores.append(float(result[3]))
		
	#normalize distances
	distsnorm = [float(i)/max(dists) for i in dists]
	scoresnorm = [float(i)/max(scores) for i in scores]
	#print()
	#print()
	#print(distsnorm)
	#print(scoresnorm)
	
	newscores = []
	for i, val in enumerate(distsnorm):
		newscores.append(mixer * distsnorm[i] + scoresnorm[i])
		results[i][3] = mixer * distsnorm[i] + scoresnorm[i]
	
	
	

	#return results
		
	#print(sorted(newscores))
		
	def sort_key0(item):
		#print(item)
		return item[3]
	def sort_key1(item):
		#print(item)
		return item[0]

	from time import sleep
	results = sorted(results, key=sort_key0, reverse=True)
	results = sorted(results, key=sort_key1, reverse=False)
	print(results)
	sleep(1)
		
	#neworder= sorted(zip(newscores,results), key=sort_key, reverse=True)
	#print(neworder)
	
	#return reranked results
	#we take away the new scores
#	neworder = [x for _,x in neworder]

	
	
	#return neworder
	return results

def writeBM25results(results, runtitle):
	f = open(runtitle, "w")
	#topic, rank, cord_id, score
	print(results)
	for result in results:
		f.write(result[0] + " Q0 " + result[2] + " " + result[1] + " " + str(result[3]) + " " + runtitle + "\n")
	f.close()

topics = readTopics("topics-rnd1.xml")
results = readAnserini("scores_covid19.txt")
writeBM25results(results, "RU-bm25")

#Now we train or load a doc2vec model
model = preparedoc2vec("./covid-doc2vec.model", prepTREC('./docids-rnd1.txt'))

results_reranked = rerank(results, model, topics, mixer=1)
writeBM25results(results_reranked, "RU-doc2vec")

#Let's try to classify each topic by making an index of the tasks
#and ranking them for each query

from constants import *
from rank_bm25 import BM25Okapi


#now let's test the manual classification

manual_classification = [2,0,3,0,3,6,3,6,7,5,4,5,0,0,0,0,6,5,5,1,0,0,1,1,1,6,6,3,3,3]

tokenized_corpus = [doc.split(" ") for doc in list_of_tasks]
bm25 = BM25Okapi(tokenized_corpus)

#wat gaat fout? 
#coronavirus response to weather changes
#coronavirus under reporting
#coronavirus quarantine
#how does coronavirus spread   - vanwege geographic
#coronavirus super spreaders
#coronavirus outside body
#

cnt = 0
cnt2 = 0
predicts = []
for topic in topics:
	tokenized_query = topic[0].split(" ")

	doc_scores = bm25.get_scores(tokenized_query)
	ind = np.argmax(doc_scores)
	print(topic[0])
	print(doc_scores)
	print("predicted " + str(ind))
	print("true " + str(topic[1]))
	print(list_of_tasks_short[ind])
	print(list_of_tasks_short[topic[1]])
	if (topic[1] == ind):
		cnt += 1
	elif ind == 7:
		cnt2 += 1
	predicts.append(ind)
		
print(cnt)
print(cnt2)

print(predicts)
print(len(predicts))