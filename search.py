#for elastic response (list of hits)
def countGTruth(results, gtruth):
	counter = 0
	for hit in results:
		if(hit.meta.id in gtruth):
			counter += 1
	return counter

#for numpy array
def countGTruthNP(results, gtruth):
	counter = 0
	for hit in results:
		if(hit[0] in gtruth):
			counter += 1
	return counter






#Let's get a proxy ground truth for the tasks
gtruths = []
for tsk in list_of_tasks_short:
	gtruths.append(getGroundtruth(tsk, 100))

# Get distances of each doc's abstract to each task
distances = docToTaskScores("docscores")



	
#So now let's see what happens if we rerank the top x documents based on distances
q = "coronavirus"
task = 2
mixer = 1
results_bm25 = query(q, 1000)
results_reranked = reranking(results_bm25, task, model, mixer)

print("\n\nQuery: " + q)
print("Task: " + list_of_tasks_short[task])

results_rerank1 = mixDistancesScores(results_bm25, distances, t, gtruths, q)

print(countGTruth(results_bm25, gtruths[t]))
print(countGTruth(results_rerank1, gtruths[t]))