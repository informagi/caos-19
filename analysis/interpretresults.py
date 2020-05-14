import numpy as np
from time import sleep

qrels = []
for line in open('qrels-rnd1.txt').readlines():
	vals = line.strip().split(" ")
	#topic, cord_uid, qrel, assessround
	qrels.append([int(vals[0]), vals[3], float(vals[4]), float(vals[1])])

qrels = np.array(qrels, dtype="O")


def pratio(vals, qrels):
	vals = np.array(vals, dtype="O")
	
#	print(vals)
	#TODO average per topic now
	#TODO average per task
	topicPs = []
	#for each topic
	for i in range(1, 31):
		#get the relevant docs for this topic (from our predictions)
		topicrels = vals[vals[:,0] == i]
		
		#get the relevant qrels for this topic (from the assessments)
		qrels_topic = qrels[qrels[:,0] == i]
		qrel_ids = [val[1] for val in qrels_topic]

		#check the top 5
		knownItems = []
		for result in topicrels[0:5]:
			#print('above')
			if(result[1] in qrel_ids):
	#			print(result) anchor
	#			print(get_qrel(result[1], qrels))
	#			sleep(1)
			
				knownItems.append(get_qrel(result[1], result[0], qrels_topic))
				#knownItems.append(list(result))
		
		#If no results were known, dno't add a value for this topic
		if(len(knownItems) > 0):
#			knownItems.append(0)
#			print('unknown')
		#print(knownItems)
		
#		print('above')
		#then check out the top 5 where we filtered out unknown qrels
		
			#avoid 0 / 2
			v = np.mean(np.array(knownItems))
			if (v < 0.01):
				topicPs.append(0)
			else:
				topicPs.append(v)
	return str(np.mean(topicPs))


def p5(vals):
	vals = np.array(vals, dtype="O")
	#TODO average per topic now
	#TODO average per task
	topicPs = []
	for i in range(1, 31):
		#test = vals[:,0 == i]
		#print(test)
		topicrels = vals[vals[:,0] == i]
		#print(topicrels[0:5,2])
		
		v=np.mean(topicrels[0:5,2])
		if (v<0.001):
			topicPs.append(0)
		else:
			topicPs.append(v)
	return str(np.mean(topicPs))
	
	#print(np.array(vals, dtype="O")[0:5,2])
	#If using np to convert a multi-type array, 
	#you have to tell it to use the original data type
	#return str(np.mean(np.array(vals, dtype="O")[0:5,2]))

def get_qrel(cord_uid, topic_id, qrels):

	topicrels = qrels[qrels[:,0] == topic_id]

	qrel_uids = [qrel[1] for qrel in topicrels]
	#print(qrel_uids.index(cord_uid))
	index = qrel_uids.index(cord_uid)
	#print(qrels[index,2])
	if(qrels[index,2] > 0):
		return 1
	else:
		return 0
	
#	return topicrels[index,2] / float(2)
	
	
	#for index, uid in qrel_uids:
	#	if 

#	qrels = np.array(qrels, dtype="O")
#	print(np.where(qrels == cord_uid))


#due to the large number of missing assessments and high
#proportion of relevant documents, there is a bias against
#atypical runs. 
#Instead of just P@5, let's see how well we do on some
#adjusted metrics to understand why results were poor
def computeMetrics(fname, qrels):
	print("Running " + fname)
	preds = []
	for line in open(fname).readlines():
		#topic, unused, cord_uid, rank, score, runname
		vals = line.strip().split(" ")
		#topic, cord_uid, score, rank
		preds.append([int(vals[0]), vals[2], float(vals[4]), int(vals[3])])
		

	knownpreds = []
	allpreds = []
#	print(qrels)


#	print(qrel_uids)
	for num, pred in enumerate(preds):
		#get qrels for the given topic
		qrels_topic = qrels[qrels[:,0] == pred[0]]
		qrel_uids = [val[1] for val in qrels_topic]	
	
		#filter all preds not in qrels
		if(pred[1] in qrel_uids):
			#add real val
			knownpreds.append([pred[0], pred[1], get_qrel(pred[1], pred[0], qrels_topic), pred[3]])
			
	print()
	print("Number of predictions")
	print(len(preds))
	print("Number of qrels in preds")
	print(len(knownpreds))
	
	print("P@5 " + p5(knownpreds))
	print("Pratio " + pratio(preds, qrels))
		
	return preds
	
preds_doc2vec = computeMetrics('RUIR-doc2vec.txt', qrels)
preds_automatic = computeMetrics('RUIR-bm25-at-exp.txt', qrels)
preds_manual = computeMetrics('RUIR-bm25-mt-exp.txt', qrels)