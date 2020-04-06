import os
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from helpers import *

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch_dsl.query import MultiMatch
from elasticsearch_dsl import Search, Q

from time import sleep
import random

import gensim
#from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 


task_1 = "What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control? Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery. Prevalence of asymptomatic shedding and transmission (e.g., particularly children). Seasonality of transmission. Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding). Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood). Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic). Natural history of the virus and shedding of it from an infected person. Implementation of diagnostics and products to improve clinical processes. Disease models, including animal models for infection, disease and transmission. Tools and studies to monitor phenotypic change and potential adaptation of the virus. Immune response and immunity. Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings. Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings. Role of the environment in transmission"
task_2 = "What do we know about COVID-19 risk factors? What have we learned from epidemiological studies? Data on potential risks factors. Smoking, pre-existing pulmonary disease. Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities. Neonates and pregnant women. Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences. Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors. Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups. Susceptibility of populations. Public health mitigation measures that could be effective for control."
task_3 = "What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface? Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time. Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged. Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over. Evidence of whether farmers are infected, and whether farmers could have played a role in the origin. Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia. Experimental infections to test host range for this pathogen. Animal host(s) and any evidence of continued spill-over to humans. Socioeconomic and behavioral risk factors for this spill-over. Sustainable risk reduction strategies"

list_of_tasks = [task_1, task_2, task_3]




def loaddocs():
	for file in tqdm(all_files):
		features = [
			file['paper_id'],
			file['metadata']['title'],
			format_authors(file['metadata']['authors']),
			format_authors(file['metadata']['authors'], 
						with_affiliation=True),
			format_body(file['abstract']),
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
	


def preparedoc2vec(fname):
	#Check if a model exists
	if(os.path.isfile(fname)):
		print("Loaded doc2vec model " + fname)
		model = Doc2Vec.load(fname)
	else:
		print("Training doc2vec model " + fname)
		
		documents = [TaggedDocument(doc[4], [i]) for i, doc in enumerate(cleaned_files)]
		print("Sanity check: is this an abstract?")
		print(documents[0])
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
#	elastic_index(indexName)
	

# query elastic, size = how many results
def query(q, size=1000):
	
	s = Search(using=es_client, index=indexName)
	
	#Theme filter is disabled
#	if(len(themeid) == 1):
		#print('Going to filter with theme ' + str(themeid))
#		s = s.filter('term', theme=int(themeid))
	s = s.query("multi_match", query = q, fields = ["title", "fulltext", "abstract"])
		
	s2 = s[0:size]
	response = s2.execute()
#	print('test here')
#	print(response)
	
#	for hit in response:
#		print(hit.theme)
	
	return response


def get_doc_vector(model, doc):
    tokens = gensim.parsing.preprocess_string(doc)
    vector = model.infer_vector(tokens)
    return vector


# Performs query, 
def reranking(results, model, task, mixer):
	taskvector = get_doc_vector(model, task)
	print(task)

	dists = []
	scores = []
	for result in results:
		dist = np.linalg.norm(taskvector-get_doc_vector(model, result['abstract']))
		dists.append(dist)
		scores.append(result.meta.score)
		
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
		
	#print(sorted(newscores))
		
	def sort_key(item):
		#print(item)
		return item[0]
		
	neworder= sorted(zip(newscores,results), key=sort_key, reverse=True)
	#print(neworder)
	
	#return reranked results
	#we take away the new scores
	neworder = [x for _,x in neworder]
	return neworder
	
def printtop(results, num):
	for i in range(0, num):
		print(results[i]['title'])
		#print(' ' + results[i]['abstract'][0:50])
		#print()



  

def wordcloud(results):
	big_chungus = []
	for result in results:
		big_chungus.append(result['title'] + ' ' + result['abstract'])
		
	#df = pd.Dataframe(big_chungus) 
	  
	comment_words = ' '
	stopwords = set(STOPWORDS) 
	  
	# iterate through the csv file 
	for val in big_chungus:#df.CONTENT:   
		# split the value 
		tokens = val.split() 
		  
		# Converts each token into lowercase 
		for i in range(len(tokens)): 
			tokens[i] = tokens[i].lower() 
			  
		for words in tokens: 
			comment_words = comment_words + words + ' '
	  
	  
	wordcloud = WordCloud(width = 800, height = 800, 
					background_color ='white', 
					stopwords = stopwords, 
					min_font_size = 30).generate(comment_words) 
	  
	# plot the WordCloud image                        
	plt.figure(figsize = (8, 8), facecolor = None) 
	plt.imshow(wordcloud, interpolation="bilinear") 
	plt.axis("off") 
	plt.tight_layout(pad = 0) 
	  
		
	

#Starting point!

#set up elastic
indexName = "3-27-4-covid"
es_client = Elasticsearch(http_compress=True)

processfiles("biorxiv_medrxiv")
#processfiles("comm_use_subset")
#processfiles("pmc_custom_license")
#processfiles("noncomm_use_subset")

#now we train or load a doc2vec model
model = preparedoc2vec("./covid-doc2vec.model")

#So now let's see how stuff is working!
results_bm25 = query("coronavirus transmission")
results_reranked = reranking(results_bm25, model, task_1, 10)

print("\n\nComparing BM25 and reranked")
print("\nOriginal")
printtop(results_bm25, 10)
print("\nReranked transmission/incubation/environment")
printtop(results_reranked, 10)

#This is an awkward way to compare... 
#Compare word cloud of top 100 results
	
wordcloud(results_bm25[0:100])
wordcloud(results_reranked[0:100])

plt.show() 	