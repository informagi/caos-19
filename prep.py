from helpers import *
from constants import *
import os
import json
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

'''
to load all/a subset of files, uncomment the lines that call process files()
to index in elastic, uncomment the elastic_index line at 111
the doc2vec model is based on the abstracts in metadata.csv - does not need all files
to rebuild it, delete it and run this file
'''


def load_docs():
    for file in tqdm(all_files):
        # Not all filetypes have abstracts! E.g. expert reviews   MIGHT crash now  and should just be abstr then
        abstr = [{'text': ''}]
        if 'abstract' in file:
            abstr = format_body(file['abstract'])
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'],
                           with_affiliation=True),
            abstr,
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)


def elastic_generator(indexname):
    for file in tqdm(cleaned_files):
        yield {
            "_index": indexname,
            "_type": "_doc",
            "_id": file[0],
            "_source": {
                "title": file[1],
                "abstract": file[4],
                "fulltext": file[5]
            }
        }


def elastic_index(indexname):
    helpers.bulk(es_client, elastic_generator(indexname))


metadatana = []


def prepare_doc2vec(fname, data):
    # Check if a model exists
    if os.path.isfile(fname):
        print("Loaded doc2vec model " + fname)
        d2v_model = Doc2Vec.load(fname)
    else:
        print("Training doc2vec model " + fname)

        # Remove items with bad abstracts
        # remove items with mising abstracts
        docs = data[~data.abstract.isnull()]
        doc_vals = docs['abstract']
        doc_vals = doc_vals.values.tolist()

        documents = [TaggedDocument(gensim.parsing.preprocess_string(doc), [i]) for i, doc in enumerate(doc_vals)]
        # print('NUMBER OF DOCS ' + str(np.array(documents)))

        # this used to be trained on the processedfiles, but abstract is available in metadata in new version
        # [TaggedDocument(doc[4], [i]) for i, doc in enumerate(cleaned_files)]
        # print("Sanity check: is this an abstract?")
        # print(documents[0])
        d2v_model = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=100, min_count=2, epochs=20, seed=42, workers=3)
        d2v_model.build_vocab(documents)
        d2v_model.train(documents, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

        d2v_model.save(fname)

    return d2v_model


# process/read/clean all files
# also for indexing the files
all_files = []
cleaned_files = []


def process_files(read_name):
    read_dir = './data/' + read_name + "/"
    file_names = os.listdir(read_dir)
    print("Number of articles retrieved from " + read_name + ":", len(file_names))

    for filename in file_names:
        filename = read_dir + filename
        file = json.load(open(filename, 'rb'))
        all_files.append(file)

    # load and clean the documents
    load_docs()

# index it in elastic
# elastic_index(indexName)


def prep_trec(fname):
    # get valid TREC ids for this round
    trec_ids = []
    f = open(fname)
    for line in f.readlines():
        if line[-1] == '\n':
            line = line[:-1]
        trec_ids.append(line)
    f.close()

    trec_metadata = pd.read_csv("./data/metadata.csv")
    # now we filter all TREC ids we don't need
    trec_metadata = trec_metadata[trec_metadata.cord_uid.isin(trec_ids)]
    # there's 33 duplicates (same paper from multiple sources)
    trec_metadata.drop_duplicates(subset='cord_uid', keep='first', inplace=True)

    return metadata


# Used to store/load scores of docs to the tasks
def doc_to_task_scores(fname):
    scores = []
    if os.path.isfile(fname):
        print("Loaded document-to-task scores from " + fname)
        for line in open(fname):
            scores.append(line.replace("\n", "").split(" "))
    else:
        print("Storing document-to-task scores in " + fname)
        f = open(fname, "a")

        for file in cleaned_files:
            newline = str(file[0])
            for index, t in enumerate(list_of_tasks):
                # If there is no abstract, we say it's very far away for now.
                # Based on assumption that these are less valuable
                # Alternative: use the last 200 words
                # print(" ".join(file[5].split(" ")[-200:]))
                dist = np.linalg.norm(task_vectors[index] - get_doc_vector(model, " ".join(file[5].split(" ")[-200:])))
                newline += " " + str(dist)

            f.write(newline + "\n")
            scores.append(newline.split(" "))

        f.close()
    return scores


def get_doc_vector(w2v_model, doc):
    tokens = gensim.parsing.preprocess_string(doc)
    vector = w2v_model.infer_vector(tokens)
    return vector


# set up elastic
indexName = "4-10-covid"
es_client = Elasticsearch(http_compress=True)

# Process the files: clean them and index them
# processfiles("biorxiv_medrxiv/pdf_json")
# processfiles("comm_use_subset/pmc_json")
# processfiles("comm_use_subset/pdf_json")
# processfiles("custom_license/pmc_json")
# processfiles("custom_license/pdf_json")
# processfiles("noncomm_use_subset/pmc_json")
# processfiles("noncomm_use_subset/pdf_json")

metadata = prep_trec('./docids-rnd1.txt')

# Now we train or load a doc2vec model
model = prepare_doc2vec("./covid-doc2vec.model", metadata)

# Now we compute the distance of each doc to each task vector, I guess
# And store that in docscores

# create a list of the taskvectors
task_vectors = []
for task in list_of_tasks:
    task_vectors.append(get_doc_vector(model, task))

# doc2task scores are not used anymore; were used to find the 1000 docs closest to a task
# docToTaskScores('docscores')
