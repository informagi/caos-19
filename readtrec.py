from prep import preparedoc2vec, prepTREC, get_doc_vector, list_of_tasks
import numpy as np
import xml.etree.ElementTree as ET
#in constants, add manually added class

#taken as output from queryclassification.py
#rnd1classes = [2,0,3,0,3,6,3,6,7,5,4,5,0,0,0,0,6,5,5,1,0,0,1,1,1,6,6,3,3,3]
rnd1classes = [2,0,3,0,3,6,3,6,7,5,4,5,0,0,0,0,3,5,5,1,0,0,1,1,6,6,6,3,3,3]

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
	abstract = metadata[metadata['cord_uid'] == cord_id]['abstract']
	return abstract.to_string()

def rerank(results, model, topics, mixer=1):
	print("reranking")
	metadata = prepTREC('./docids-rnd1.txt')

	
	dists = []
	scores = []
	for result in results:
		sha = topics[int(result[0]) - 1]
		taskvector = get_doc_vector(model, sha[0])
		dist = np.linalg.norm(taskvector-get_doc_vector(model, getAbstract(result[2], metadata)))
		dists.append(dist)
		scores.append(float(result[3]))
		
	#normalize distances
	distsnorm = [float(i)/max(dists) for i in dists]
	scoresnorm = [float(i)/max(scores) for i in scores]
	
	newscores = []
	for i, val in enumerate(distsnorm):
		newscores.append(mixer * distsnorm[i] + scoresnorm[i])
		results[i][3] = mixer * distsnorm[i] + scoresnorm[i]
	
	#Some ugly/quick sorting
	def sort_key0(item):
		return item[3]
	def sort_key1(item):
		return item[0]

	results = sorted(results, key=sort_key0, reverse=True)
	results = sorted(results, key=sort_key1, reverse=False)
	
	#return neworder
	return results

#prepare results in submission format
def writeBM25results(results, runtitle):
	f = open(runtitle, "w")
	#topic, rank, cord_id, score
	for result in results:
		f.write(result[0] + " Q0 " + result[2] + " " + result[1] + " " + str(result[3]) + " " + runtitle + "\n")
	f.close()

topics = readTopics("topics-rnd1.xml")
results = readAnserini("anserini_bm25.txt")
writeBM25results(results, "RUIR-bm25")

#Now we train or load a doc2vec model
model = preparedoc2vec("./covid-doc2vec.model", prepTREC('./docids-rnd1.txt'))

results_reranked = rerank(results, model, topics, mixer=1)
writeBM25results(results_reranked, "RUIR-doc2vec")