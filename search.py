from constants import *
from prep import *
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

import numpy as np


# for elastic response (list of hits)
def countGTruth(results, gtruth):
    counter = 0
    for hit in results:
        if (hit.meta.id in gtruth):
            counter += 1
    return counter


# for numpy array
def countGTruthNP(results, gtruth):
    counter = 0
    for hit in results:
        if (hit[0] in gtruth):
            counter += 1
    return counter


def compareSearch(results1, results2, title1="BM25", title2="fulltext", numresults=10):
    print("Comparing " + title1 + " and " + title2)
    print("\n" + title1)
    printtop(results_bm25, numresults)
    print("\n" + title2)
    printtop(results_reranked, numresults)


def getGroundtruth(q, size=1000):
    results = query(q, size)
    ids = []
    for result in results:
        #		print(result.meta.id)
        ids.append(result.meta.id)
    return ids


# first attempt at reranking: get query, rerank top 1000 based on how close to the task they are in doc2vec space
# Performs query, 
def reranking(results, task, model, mixer=1):
    taskvector = get_doc_vector(model, list_of_tasks[task])
    # print(task)

    dists = []
    scores = []
    for result in results:
        dist = np.linalg.norm(taskvector - get_doc_vector(model, result['abstract']))
        dists.append(dist)
        scores.append(result.meta.score)

    # normalize distances
    distsnorm = [float(i) / max(dists) for i in dists]
    scoresnorm = [float(i) / max(scores) for i in scores]
    # print()
    # print()
    # print(distsnorm)
    # print(scoresnorm)

    newscores = []
    for i, val in enumerate(distsnorm):
        newscores.append(mixer * distsnorm[i] + scoresnorm[i])

    # print(sorted(newscores))

    def sort_key(item):
        # print(item)
        return item[0]

    neworder = sorted(zip(newscores, results), key=sort_key, reverse=True)
    # print(neworder)

    # return reranked results
    # we take away the new scores
    neworder = [x for _, x in neworder]
    return neworder


def getDist(fileid, distances, task):
    # Should be possible to look up by value? but I'm lazy and performance doesn't change much
    #	print(distances[1])
    #	sleep(10)
    for dist in distances:
        if fileid == dist[0]:
            return float(dist[task + 1])
    else:
        # Not in top 1000 distances
        return -1


# query elastic, size = how many results
def query(q, size=1000):
    s = Search(using=es_client, index=indexName)

    # Theme filter is disabled
    #	if(len(themeid) == 1):
    # print('Going to filter with theme ' + str(themeid))
    #		s = s.filter('term', theme=int(themeid))
    s = s.query("multi_match", query=q, fields=["title", "fulltext", "abstract"]).highlight('fulltext',
                                                                                            fragment_size=100)

    s2 = s[0:size]
    response = s2.execute()
    #	print('test here')
    #	print(response)

    #	for hit in response:
    #		print(hit.theme)

    return response


# Should probably collect all id's before querying, but this works for now
def get_doc_score(q, fileid):
    s = Search(using=es_client, index=indexName)
    #	print(fileid)
    s = s.filter("term", _id=fileid).query("multi_match", query=q,
                                           fields=["title", "fulltext", "abstract"])  # .query("id", id = fileid)
    response = s.execute()
    # print(response)
    # print(fileid)
    # sleep(1)

    # If we retrieved something (i.e. our keywords appeared), add that
    if (hasattr(response, 'hits')):
        # should only be one.. so return the first
        for hit in response:
            return hit.meta.score
    # print("Could not find in index:")
    # print(fileid)
    return -1


# Second attempt at reranking: felt the previous wasn't explorative enough/didn't change enough by reranking1
# The same, but also include the top 1000 docs closest to the given task
def mixDistancesScores(results, distances, task, gtruths, q, mixer=1):
    distances = np.array(distances)
    resultlist = []  # will be an array with three columns: id, score, distance, mixed_score
    maxscore = -1
    minscore = 999
    maxdist = 0
    mindist = 99999

    # If task is not set, we should predict it.
    if task == -1:
        t = predictClass(results, gtruths)

    # First append the ones we found through search
    for hit in results:
        # Find the distance of this hit to this task.
        d = getDist(hit.meta.id, distances, task)

        resultlist.append([hit.meta.id, hit.meta.score, d, -1])
        if hit.meta.score > maxscore:
            maxscore = hit.meta.score
        if hit.meta.score < minscore and hit.meta.score >= 0:
            minscore = hit.meta.score
        # print(d)
        # print(maxdist)
        if d > maxdist and d <= 1:
            maxdist = d
        if d < mindist and d >= 0:
            mindist = d

    # Then append the 1000 closest to this task.

    # First find the top 1000
    print(distances.shape)
    distances = distances[distances[:, task + 1].argsort()]
    distances = distances[0:1000]

    # Then add them
    for dist in distances:
        # print(float(dist[task+1]))
        # for each file, find bm25 score to this fileid
        s = get_doc_score(q, dist[0])
        #		print(s)

        resultlist.append([dist[0], s, float(dist[task + 1]), -1])

        # note: we are looking for the minimum dist..
        if float(dist[task + 1]) > maxdist and float(dist[task + 1]) <= 1:
            maxdist = float(dist[task + 1])
        if float(dist[task + 1]) < mindist and float(dist[task + 1]) >= 0:
            mindist = float(dist[task + 1])
        if s > maxscore:
            maxscore = s
        if s < minscore and s >= 0:
            minscore = s

    #		print([dist[0],s,float(dist[t+2]),-1])
    # print(resultlist)

    # TODO Make it a set

    #	print("hello")
    #	print(mindist)
    #	print(maxdist)
    #	print(minscore)
    #	print(maxscore)
    countmissing = 0
    # Normalize. Done so verbose for debugging
    for index, result in enumerate(resultlist):
        # If it didn't appear in the top results (i.e. is negative) of a task, set the distance to max
        if (float(result[2]) < 0):
            resultlist[index][2] = 1
        # Otherwise normalize it
        else:
            resultlist[index][2] = (float(resultlist[index][2]) - mindist) / (float(maxdist) - mindist)

        # If the score was not set (ie not found by query), set to minimum score possible
        if (float(result[1]) < 0):
            countmissing += 1
            resultlist[index][1] = 0
        else:
            # If it was not , normalize
            resultlist[index][1] = (float(resultlist[index][1]) - minscore) / (float(maxscore) - minscore)

        # combine the scores
        #	print(resultlist[index][1])
        #	print(resultlist[index][2])
        resultlist[index][3] = resultlist[index][1] + mixer * (1 - resultlist[index][2])

    # Take the minimum because due to float computation errors we can get values over the max distance
    # TODO something is terribly wrong. I should be getting distances ranging from 0 to 1, with the lower ones scoring higher overall.
    # TODO for some reason the filter by id cannot find anything.

    #			resultlist[index][2] = float(resultlist[index][2]) / float(maxdist)
    print('missing')
    print(countmissing)
    #	print(maxscore)
    #	print(maxdist)
    # Compute the new scores
    #	for index, result in enumerate(resultlist):
    # Don't forget to normalize scores
    #		print(result)
    #		print(result)
    #		print(maxscore)
    #		print((result[1] / maxscore))
    #		sleep(1)
    #	resultlist[index][1] = (result[1] / maxscore)
    #	resultlist[index][2] = (float(result[2]) / float(maxdist))

    # Take the average of the score and the inverted distance
    #		resultlist[index][3] = result[1] + mixer * (1 - result[2])

    f = open('testme', 'w')
    for item in resultlist:
        f.write(str(item) + "\n")
    f.close()

    # Reorder the results
    resultlist = np.array(resultlist)
    resultlist = resultlist[resultlist[:, 3].argsort()]

    print(resultlist)  # [0:5][3]

    return resultlist


# NOTE the highlighter returns multiple sentencess
def printtop(results, num):
    for i in range(0, num):
        print("\n" + str(i))
        print("TITLE  " + results[i]['title'])
        if (hasattr(results[i].meta, "highlight")):
            if (hasattr(results[i].meta.highlight, "abstract")):
                print("ABSTRACT  " + results[i].meta.highlight.abstract[0])
            if (hasattr(results[i].meta.highlight, "fulltext")):
                print("FULLTEXT  " + results[i].meta.highlight.fulltext[0])


#		print(results[i].meta.highlight.abstract)
#		print(results[i].meta.highlight.fulltext)
#		for field in results[i].meta.highlight:
#			print(field)
#		if(hasattr(results[i].highlight, 'highlight')):
#			print(results[i].meta)
# print(' ' + results[i]['abstract'][0:50])
# print()


# set up elastic
indexName = "4-10-covid"
es_client = Elasticsearch(http_compress=True)

# Let's get a (kind ugly) proxy ground truth for the tasks
gtruths = []
for tsk in list_of_tasks_short:
    gtruths.append(getGroundtruth(tsk, 100))

# Get distances of each doc's abstract to each task
distances = docToTaskScores("docscores")

# So now let's see what happens if we rerank the top x documents based on distances
q = "coronavirus"
# Task: risk_factors
task = 1
mixer = 1
results_bm25 = query(q, 1000)
results_rerank1 = reranking(results_bm25, task, model, mixer)

print("\n\nQuery: " + q)
print("Task: " + list_of_tasks_short[task])

results_rerank2 = mixDistancesScores(results_bm25, distances, task, gtruths, q)

print(countGTruth(results_bm25, gtruths[task]))
print(countGTruth(results_rerank1, gtruths[task]))
print(countGTruthNP(results_rerank2, gtruths[task]))
compareSearch(results_bm25, mixedresults)
