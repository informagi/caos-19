from helpers import *
import os
import json
import numpy as np
from tqdm.notebook import tqdm
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd 


def loaddocs():
	for file in tqdm(all_files):
		#Not all filetypes have abstracts! E.g. expert reviews
		abstr = [{'text':''}]
		if 'abstract' in file:
			abstr = file['abstract']
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
			"_id" : file[0],
			"_source": {
				"title":file[1],
				"abstract":file[4],
				"fulltext":file[5]
			}
		}


def elastic_index(indexname):
	helpers.bulk(es_client, elastic_generator(indexname))
	


def preparedoc2vec(fname, data):
	#Check if a model exists
	if(os.path.isfile(fname)):
		print("Loaded doc2vec model " + fname)
		model = Doc2Vec.load(fname)
	else:
		print("Training doc2vec model " + fname)
		
		documents = data['abstract'].values.tolist()
		#this used to be trained on the processedfiles, but abstract is available in metadata in new version
		#[TaggedDocument(doc[4], [i]) for i, doc in enumerate(cleaned_files)]
		#print("Sanity check: is this an abstract?")
		#print(documents[0])
		model = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=100, min_count=2, epochs=20, seed=42, workers=3)
		model.build_vocab(documents)
		model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
		
		model.save(fname)

	return model


#process/read/clean all files
#also for indexing the files
all_files = []
cleaned_files = []
def processfiles(readname):
	readdir = './data/' + readname + "/"
	filenames = os.listdir(readdir)
	print("Number of articles retrieved from " + readname + ":", len(filenames))

	for filename in filenames:
		filename = readdir + filename
		file = json.load(open(filename, 'rb'))
		all_files.append(file)
		
		
	#load and clean the documents
	loaddocs()
	
	#index it in elastic
	elastic_index(indexName)
	
def prepTREC(fname):
	#get valid TREC ids for this round
	TRECids = []
	f = open(fname)
	for line in f.readlines():
		if line[-1] == '\n':
			line = line[:-1]
		TRECids.append(line)
	f.close()
	
	metadata = pd.read_csv("./data/metadata.csv")
	#now we filter all TREC ids we don't need
	#print(np.array(metadata).shape)
	#print(np.array(TRECids).shape)
	metadata = metadata[metadata.cord_uid.isin(TRECids)]
	#there's 33 duplicates (same paper from multiple sources)
	metadata.drop_duplicates(subset='cord_uid', keep='first', inplace=True)
	
	return metadata

#set up elastic
indexName = "4-10-covid"
es_client = Elasticsearch(http_compress=True)

#Process the files: clean them and index them
#processfiles("biorxiv_medrxiv/pdf_json")
#processfiles("comm_use_subset/pmc_json")
#processfiles("comm_use_subset/pdf_json")
#processfiles("custom_license/pmc_json")
#processfiles("custom_license/pdf_json")
#processfiles("noncomm_use_subset/pmc_json")
#processfiles("noncomm_use_subset/pdf_json")

metadata = prepTREC('./docids-rnd1.txt')

#Now we train or load a doc2vec model
model = preparedoc2vec("./covid-doc2vec.model", metadata)