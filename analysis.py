import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS

from constants import *
from prep import preparedoc2vec, prep_trec, get_doc_vector


def wordcloud(results):
    big_chungus = []
    for result in results:
        big_chungus.append(result['title'] + ' ' + result['abstract'])

    comment_words = ' '
    stopwords = set(STOPWORDS)

    # iterate through the csv file
    for val in big_chungus:
        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        for words in tokens:
            comment_words = comment_words + words + ' '

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=30).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)


def printdistances(kmeans, taskvectors):
    print('task cluster1 cluster2 etc')
    for index, task in enumerate(taskvectors):
        print(index)
        line = ""
        for cluster in kmeans.cluster_centers_:
            print(cluster)
            print(task)
            d = np.linalg.norm(task - cluster)
            line += '  ' + str(d)
        print(line)


def avgDistanceToCluster(kmeans, abstract_vectors):
    sumdistances = []
    numdistances = 0

    for index, cluster in enumerate(kmeans.cluster_centers_):
        sumdistances.append(0)

    for vector in abstract_vectors:
        listd = []
        for cluster in kmeans.cluster_centers_:
            listd.append(np.linalg.norm(task - cluster))

        listd = sorted(listd)

        # add it to the list
        for index, value in enumerate(sumdistances):
            sumdistances[index] += listd[index]

    avgdistances = []
    # Finally, take the avgs
    for index, value in enumerate(sumdistances):
        avgdistances.append(sumdistances[index])
        print(sumdistances[index])

    return avgdistances


# printRiskFactors(riskdata, trainshape)

# def printRiskFactors():
#	riskdata = np.concatenate((model.docvecs.vectors_docs, riskvectors))
#	print(riskdata.shape)

#	doc2vec_tsne = tsne.fit_transform(riskdata)
#	print(np.array(model.docvecs.vectors_docs).shape)
#	print(np.array(doc2vec_tsne).shape)

#	fig = go.Figure()
#	fig.add_trace(go.Scatter(x=doc2vec_tsne[:trainshape,0], y=doc2vec_tsne[:trainshape,1],mode='markers'))
#	fig.add_trace(go.Scatter(x=doc2vec_tsne[trainshape:,0], y=doc2vec_tsne[trainshape:,1],mode='markers'))
#	fig.show()

# def printClusters(abstract_vectors, labels, k=13):
#	print(np.array(abstract_vectors).shape)

# the arrays we will store the k=13 clusters
#	clustervectors = []
#	for i in range(k):
#		clustervectors.append([])

# adding the risk factors to the training set
#	for index, vector in enumerate(abstract_vectors):
#		clustervectors[labels[index]].append(index)

#	print(np.array(clustervectors).shape)
#	doc2vec_tsne = tsne.fit_transform(abstract_vectors)

#	fig = go.Figure()
#	for cluster in clustervectors:


#		fig.add_trace(go.Scatter(x=doc2vec_tsne[cluster,0], y=doc2vec_tsne[cluster,1],mode='markers'))
#	fig.show()


# print clusters and tasks
# abstract_vectors is all the abstracts, labels is the label for each cluster
def printClusterTasks(abstract_vectors, labels, k=13):
    clustervectors = []
    for i in range(k):
        clustervectors.append([])

    # adding the risk factors to the training set
    for index, vector in enumerate(abstract_vectors):
        clustervectors[labels[index]].append(index)

    abstract_length = np.array(abstract_vectors).shape[0]

    riskdata = riskvectors
    risklen = np.array(riskdata).shape[0]

    fig = go.Figure()

    # fig.add_trace(go.Scatter(x=doc2vec_tsne[:trainshape,0], y=doc2vec_tsne[:trainshape,1],mode='markers'))

    alldata = np.concatenate((abstract_vectors, taskvectors))
    doc2vec_tsne = tsne.fit_transform(alldata)

    for cluster in clustervectors:
        fig.add_trace(go.Scatter(x=doc2vec_tsne[cluster, 0], y=doc2vec_tsne[cluster, 1], mode='markers'))

    # print tasks
    fig.add_trace(
        go.Scatter(x=doc2vec_tsne[abstract_length:, 0], y=doc2vec_tsne[abstract_length:, 1], mode='lines+markers'))

    fig.show()


# Add the riskvectors to the visualization. How do they relate to the other documents?
def printTasksRisks(abstract_vectors, labels, k=13):
    # the arrays we will store the k=13 clusters
    clustervectors = []
    for i in range(k):
        clustervectors.append([])

    # adding the risk factors to the training set
    for index, vector in enumerate(abstract_vectors):
        clustervectors[labels[index]].append(index)

    abstract_length = np.array(abstract_vectors).shape[0]

    riskdata = riskvectors
    risklen = np.array(riskdata).shape[0]

    fig = go.Figure()

    # fig.add_trace(go.Scatter(x=doc2vec_tsne[:trainshape,0], y=doc2vec_tsne[:trainshape,1],mode='markers'))

    alldata = np.concatenate((abstract_vectors, taskvectors, riskdata))
    print(np.array(clustervectors).shape)
    doc2vec_tsne = tsne.fit_transform(alldata)

    for cluster in clustervectors:
        fig.add_trace(go.Scatter(x=doc2vec_tsne[cluster, 0], y=doc2vec_tsne[cluster, 1], mode='markers'))

    # print riskdata
    fig.add_trace(go.Scatter(x=doc2vec_tsne[abstract_length:abstract_length + risklen, 0],
                             y=doc2vec_tsne[abstract_length:abstract_length + risklen, 1], mode='markers'))
    # print tasks
    fig.add_trace(
        go.Scatter(x=doc2vec_tsne[abstract_length + risklen:, 0], y=doc2vec_tsne[abstract_length + risklen:, 1],
                   mode='lines+markers'))

    # print risktask
    fig.add_trace(go.Scatter(x=doc2vec_tsne[abstract_length + risklen + 1:abstract_length + risklen + 2, 0],
                             y=doc2vec_tsne[abstract_length + risklen + 1:abstract_length + risklen + 2, 1],
                             mode='lines+markers'))

    fig.show()


def printSNE1(data):
    doc2vec_tsne = tsne.fit_transform(data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=doc2vec_tsne[:, 0], y=doc2vec_tsne[:, 1], mode='markers'))
    fig.show()


# print SNE with two document sets
# data[:trainshape] and data[trainshape:]
def printSNE2(data, trainshape):
    # lets get the indices of the risk vectors

    #	risk has ..
    #	metadata.sha.isin()

    #	print(riskfactors[1][-30:])
    #	print(metadata['abstract'][1][-30:])

    rsk = pd.DataFrame(riskfactors_old).dropna(how='any')  # .values.tolist()

    # drop na's; follow approach in prep.py
    metadatana = metadata[~metadata.abstract.isin(["Unknown", "unknown", ""])]
    metadatana = metadatana[~metadatana.abstract.isnull()]

    # print(rsk[0])
    # print(rsk[1][:2])
    # metadatana = pd.DataFrame(metadatana)
    # overlap = metadatana.sha.isin(rsk[0][:])
    # print(overlap.value_counts())

    # print('test shapes')
    # print(np.array(overlap).shape)
    # print(np.array(data).shape)

    #	data = concatenate
    doc2vec_tsne = tsne.fit_transform(data)

    fig = go.Figure()

    # fig.add_trace(go.Scatter(x=doc2vec_tsne[~overlap,0], y=doc2vec_tsne[~overlap:,1],mode='markers'))
    # fig.add_trace(go.Scatter(x=doc2vec_tsne[overlap,0], y=doc2vec_tsne[overlap:,1],mode='markers'))

    fig.add_trace(go.Scatter(x=doc2vec_tsne[:trainshape, 0], y=doc2vec_tsne[:trainshape, 1], mode='markers'))
    fig.add_trace(go.Scatter(x=doc2vec_tsne[trainshape:, 0], y=doc2vec_tsne[trainshape:, 1], mode='markers'))
    fig.show()


def analyze6():
    print(np.array(trace6).shape)

    kmeans = KMeans(init='k-means++', max_iter=100, random_state=42)

    # Elbow method to figure out what we should set K to
    # visualizer = KElbowVisualizer(kmeans, k=(2, 32))
    # visualizer.fit(trace6)
    # visualizer.show()

    # k = 24 from elbow analysis
    kmeans6 = KMeans(n_clusters=24, init='k-means++', max_iter=100, random_state=42)
    labels6 = kmeans6.fit_predict(trace6)

    # printSNE2(np.concatenate((trace6, riskvectors)), np.array(trace6).shape[0])
    #	printClusters(trace6, labels6, 24)
    printTasksRisks(trace6, labels6, 24)


def getGroundtruth(q, size=1000):
    results = query(q, size)
    ids = []
    for result in results:
        #		print(result.meta.id)
        ids.append(result.meta.id)
    return ids


def getRisks(filename):
    abstracts = []
    fp = open(filename, 'r', encoding='utf-8')
    #	header = fp.readline()
    for line in fp.readlines():
        # print((line.split(",")[1:]))
        # print(",".join(line.split(",")[1:]))
        abstracts.append([line.split(",")[0], ",".join(line.split(",")[1:])])

    fp.close()
    return abstracts


def loadFacetDocs(filename):
    abstracts = []
    fp = open(filename, 'r', encoding='utf-8')
    #	header = fp.readline()
    for line in fp.readlines():
        #		strline = line.split(",")   [7]
        abstracts.append(line)

    fp.close()
    return abstracts


# Code execution starts here


# wordcloud(results_bm25[0:100])
# wordcloud(results_reranked[0:100])

metadata = prep_trec('./docids-rnd1.txt')
# Now we load the doc2vec model
model = preparedoc2vec("./covid-doc2vec.model", metadata)

# create a list of the taskvectors
taskvectors = []
for task in list_of_tasks:
    taskvectors.append(get_doc_vector(model, task))

print("Starting clustering")
# Let's do a cluster analysis: do we recognize these in the task model?
# based on https://www.kaggle.com/luisblanche/cord-19-use-doc2vec-to-match-articles-to-tasks#5.-Clustering-and-visualisation
abstract_vectors = model.docvecs.vectors_docs
kmeans = KMeans(init='k-means++', max_iter=100, random_state=42)

# Elbow method to figure out what we should set K to
# visualizer = KElbowVisualizer(kmeans, k=(2, 32))
# visualizer.fit(abstract_vectors)
# visualizer.show()

# k = 13 from elbow
k = 13
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, random_state=42)
labels = kmeans.fit_predict(abstract_vectors)

# load risk factor docs
riskfactors_old = loadFacetDocs('./data/risk_abstracts.csv')
riskfactors = getRisks('./data/risk_sha.csv')

riskvectors = []
for factor in riskfactors:
    riskvectors.append(get_doc_vector(model, factor[1]))

print('start plotting')
# perplexity of 5 and learning rate of 500 gives good results
tsne = TSNE(n_components=2, perplexity=5, learning_rate=500)
num_abstract = np.array(model.docvecs.vectors_docs).shape[0]

# printSNE1(model.docvecs.vectors_docs)
# printSNE2(model.docvecs.vectors_docs, num_abstract)
# printSNE2(np.concatenate((model.docvecs.vectors_docs, riskvectors)), num_abstract)
# printClusterTasks(abstract_vectors, labels, k)
printTasksRisks(abstract_vectors, labels, k)

# print("Check dit")
# print(abstract_vectors[0:2])
# print(list_of_tasks[0:2])
# print(riskfactors[0:2])


# So get all documents in the golden cluster
# print('Analysis of trace6')
# trace6 = abstract_vectors[labels==6]

# analyze6()


# Let's try another ranking method

# Now what's the avg distance between    riskvectors and task descriptions
# avg distance     my rerank on riskv's   and riskvectors
dists = []
for vec in riskvectors:
    dists.append(np.linalg.norm(vec - taskvectors[1]))  # get_doc_vector(model, file[4])

print("Avg and var dist riskvectors to risk task")
print(np.mean(dists))
print(np.var(dists))

dists = []
# counter = 0
for vec in abstract_vectors:
    dists.append(np.linalg.norm(vec - taskvectors[1]))  # get_doc_vector(model, file[4])
#	counter += 1

print("Avg and var dist all vectors to risk task")
print(np.mean(dists))
print(np.var(dists))

# get average risk vector
avgrisk = np.average(riskvectors, axis=0)
print("AVerage risk vector, distance to risk task vector")
print(avgrisk)
print(avgrisk.shape)
print(np.linalg.norm(avgrisk - taskvectors[1]))

# get avg distance from the avgrisk
dists = []
for vec in riskvectors:
    dists.append(np.linalg.norm(vec - avgrisk))  # get_doc_vector(model, file[4])

print("Avg and var dist all vectors to risk task")
print(np.mean(dists))
print(np.var(dists))
