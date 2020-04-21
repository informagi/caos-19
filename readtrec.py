from prep import preparedoc2vec, prepTREC, get_doc_vector, list_of_tasks
import numpy as np
import xml.etree.ElementTree as ET
#in constants, add manually added class

rnd1classes = [2,0,3,0,3,7,3,7,6,5,4,5,0,0,0,0,7,5,5,1,0,0,1,1,1,7,7,3,3,3]
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
		
	return results
		
	#print(sorted(newscores))
		
#	def sort_key(item):
		#print(item)
#		return item[0]
		
	#neworder= sorted(zip(newscores,results), key=sort_key, reverse=True)
	#print(neworder)
	
	#return reranked results
	#we take away the new scores
#	neworder = [x for _,x in neworder]

	
	
	#return neworder

def writeBM25results(results, runtitle):
	f = open(runtitle, "w")
	#topic, rank, cord_id, score
	print(results)
	for result in results:
		f.write(result[0] + " Q0 " + result[2] + " " + result[1] + " " + str(result[3]) + " " + runtitle + "\n")
	f.close()

topics = readTopics("topics-rnd1.xml")
print(topics)
results = readAnserini("scores_covid19.txt")
writeBM25results(results, "NL-RU-bm25")
print(len(results))

#Now we train or load a doc2vec model
model = preparedoc2vec("./covid-doc2vec.model", prepTREC('./docids-rnd1.txt'))
results_reranked = rerank(results, model, topics, mixer=1)

#print(results_reranked)
writeBM25results(results_reranked, "NL-RU-doc2vec")