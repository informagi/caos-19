from prep import preparedoc2vec, prep_trec, get_doc_vector
import numpy as np
import xml.etree.ElementTree as Et

# in constants, add manually added class

# taken as output from queryclassification.py
rnd1classes = [2, 0, 3, 0, 3, 6, 3, 6, 7, 5, 4, 5, 0, 0, 0, 0, 6, 5, 5, 1, 0, 0, 1, 1, 1, 6, 6, 3, 3, 3]


def read_topics(fname):
    root = Et.parse(fname).getroot()
    covid_topics = []
    for num, topic in enumerate(root):
        # print(topic[0].text) #query
        covid_topics.append([topic[0].text, rnd1classes[num]])
    # print(topic[1].text) #question
    # print(topic[2].text) #narrative
    return covid_topics


def read_anserini(file_name):
    res = []
    f = open(file_name)
    f.readline()
    for line in f.readlines():
        vals = line.strip().split(",")
        # topic, rank, cord_id, score
        res.append([vals[0], vals[1], vals[2], vals[3]])
    return res


def get_abstract(cord_id, metadata):
    abstract = metadata[metadata['cord_uid'] == cord_id]['abstract']
    return abstract.to_string()


def rerank(initial_results, w2v_model, covid_topics, mixer=1):
    print("reranking")
    metadata = prep_trec('./docids-rnd1.txt')

    dists = []
    scores = []
    for result in initial_results:
        sha = covid_topics[int(result[0]) - 1]
        taskvector = get_doc_vector(w2v_model, sha[0])
        dist = np.linalg.norm(taskvector - get_doc_vector(w2v_model, get_abstract(result[2], metadata)))
        dists.append(dist)
        scores.append(float(result[3]))

    # normalize distances
    distsnorm = [float(i) / max(dists) for i in dists]
    scoresnorm = [float(i) / max(scores) for i in scores]

    newscores = []
    for i, val in enumerate(distsnorm):
        newscores.append(mixer * distsnorm[i] + scoresnorm[i])
        initial_results[i][3] = mixer * distsnorm[i] + scoresnorm[i]

    # Some ugly/quick sorting
    def sort_key0(item):
        return item[3]

    def sort_key1(item):
        return item[0]

    initial_results = sorted(initial_results, key=sort_key0, reverse=True)
    initial_results = sorted(initial_results, key=sort_key1, reverse=False)

    # return neworder
    return initial_results


# prepare results in submission format
def write_bm25_results(bm25_results, runtitle):
    f = open(runtitle, "w")
    # topic, rank, cord_id, score
    for result in bm25_results:
        f.write(result[0] + " Q0 " + result[2] + " " + result[1] + " " + str(result[3]) + " " + runtitle + "\n")
    f.close()


topics = read_topics("topics-rnd1.xml")
results = read_anserini("anserini_bm25.txt")
write_bm25_results(results, "RUIR-bm25")

# Now we train or load a doc2vec model
model = preparedoc2vec("./covid-doc2vec.model", prep_trec('./docids-rnd1.txt'))

results_reranked = rerank(results, model, topics, mixer=1)
write_bm25_results(results_reranked, "RUIR-doc2vec")
