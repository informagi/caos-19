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

task_1_short = "transmission incubation environment"
task_1 = "What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control? Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery. Prevalence of asymptomatic shedding and transmission (e.g., particularly children). Seasonality of transmission. Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding). Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood). Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic). Natural history of the virus and shedding of it from an infected person. Implementation of diagnostics and products to improve clinical processes. Disease models, including animal models for infection, disease and transmission. Tools and studies to monitor phenotypic change and potential adaptation of the virus. Immune response and immunity. Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings. Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings. Role of the environment in transmission"
task_2_short = "risk factors"
task_2 = "What do we know about COVID-19 risk factors? What have we learned from epidemiological studies? Data on potential risks factors. Smoking, pre-existing pulmonary disease. Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities. Neonates and pregnant women. Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences. Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors. Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups. Susceptibility of populations. Public health mitigation measures that could be effective for control."
task_3_short = "genetics origin evolution"
task_3 = "What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface? Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time. Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged. Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over. Evidence of whether farmers are infected, and whether farmers could have played a role in the origin. Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia. Experimental infections to test host range for this pathogen. Animal host(s) and any evidence of continued spill-over to humans. Socioeconomic and behavioral risk factors for this spill-over. Sustainable risk reduction strategies"
task_4_short = "vaccines therapeutics"
task_4 = "What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics? Effectiveness of drugs being developed and tried to treat COVID-19 patients. Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication. Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients. Exploration of use of best animal models and their predictive value for a human vaccine. Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents. Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need. Efforts targeted at a universal coronavirus vaccine. Efforts to develop animal models and standardize challenge studies. Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers. Approaches to evaluate risk for enhanced disease after vaccination. Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models in conjunction with therapeutics."
task_5_short = "medical care"
task_5 = "What has been published about medical care? What has been published concerning surge capacity and nursing homes? What has been published concerning efforts to inform allocation of scarce resources? What do we know about personal protective equipment? What has been published concerning alternative methods to advise on disease management? What has been published concerning processes of care? What do we know about the clinical characterization and management of the virus? Resources to support skilled nursing facilities and long term care facilities. Mobilization of surge medical staff to address shortages in overwhelmed communities. Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies. Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients. Outcomes data for COVID-19 after mechanical ventilation adjusted for age. Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level. Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks. Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries. Guidance on the simple things people can do at home to take care of sick people and manage disease. Oral medications that might potentially work. Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually. Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes. Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials. Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials. Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen). "
task_6_short = "effectiveness non pharmaceutical interventions"
task_6 = "What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions? Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases. Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments. Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches. Methods to control the spread in communities, barriers to compliance and how these vary among different populations. Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status. Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs. Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high). Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."
task_7_short = "diagnostics surveillance"
task_7 = "What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)? How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs). Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms. Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues. National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public). Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy. Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded). Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices. Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes. Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling. Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions. Policies and protocols for screening and testing. Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents. Technology roadmap for diagnostics. Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment. New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases. Coupling genomics and diagnostic testing on a large scale. Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant. Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional. One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."
task_8_short = "geographic spread"
task_8 = "At the time of writing, COVID-19 has spread to at least 114 countries. With viral flu, there are often geographic variations in how the disease will spread and if there are different variations of the virus in different areas. We’d like to explore what the literature and data say about this through this Task. Are there geographic variations in the rate of COVID-19 spread? Are there geographic variations in the mortality rate of COVID-19? Is there any evidence to suggest geographic based virus mutations?"
task_9_short = "ethical social considerations"
task_9 = "What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response? Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019. Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight. Efforts to support sustained education, access, and capacity building in the area of ethics. Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences. Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures). Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed. Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media."
task_10_short = "information sharing collaboration"
task_10 = "What has been published about information sharing and inter-sectoral collaboration? What has been published about data standards and nomenclature? What has been published about governmental public health? What do we know about risk communication? What has been published about communicating with high-risk populations? What has been published to clarify community measures? What has been published about equity considerations and problems of inequity? Methods for coordinating data-gathering with standardized nomenclature. Sharing response information among planners, providers, and others. Understanding and mitigating barriers to information-sharing. How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic). Integration of federal/state/local public health surveillance systems. Value of investments in baseline public health response infrastructure preparedness. Modes of communicating with target high-risk populations (elderly, health care workers). Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too). Communication that indicates potential risk of disease to all population groups. Misunderstanding around containment and mitigation. Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment. Measures to reach marginalized and disadvantaged populations.Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities. Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment. Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care."


list_of_tasks = [task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9, task_10]
list_of_tasks_short = [task_1_short, task_2_short, task_3_short, task_4_short, task_5_short, task_6_short, task_7_short, task_8_short, task_9_short, task_10_short]



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
	readdir = './covid/data/' + readname + "/"
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
	s = s.query("multi_match", query = q, fields = ["title", "fulltext", "abstract"]).highlight('fulltext', fragment_size=100)
		
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

#Should probably collect all id's before querying, but this works for now
def get_doc_score(q, fileid):
	s = Search(using=es_client, index=indexName)
#	print(fileid)
	s = s.filter("term", _id=fileid).query("multi_match", query = q, fields = ["title", "fulltext", "abstract"])#.query("id", id = fileid)
	response = s.execute()
	#print(response)
	#print(fileid)
	#sleep(1)
	
	#If we retrieved something (i.e. our keywords appeared), add that
	print(response)
	sleep(1)
	if(hasattr(response, 'hit')):
		for field in response.hit:
			print(field)
		return response[0].meta.score
	return -1


#def getTaskDocs()
#	Opties
#		* Sla in Elastic op score voor elke task, kijk hoe je daar reranked
#		* Sla lokaal op wat de top 1000 docs zijn voor elke taak, en combineer daar. voor deze gaan we


def docToTaskScores(fname):
	scores = []
	if(os.path.isfile(fname)):
		print("Loaded document-to-task scores from " + fname)
		for line in open(fname):
			scores.append(line.replace("\n", "").split(" "))
#		print(scores)
	else:
		print("Storing document-to-task scores in " + fname)
		f = open(fname, "a")
		
		#create a list of the taskvectors, just to save some computing time
		taskvectors = []
		for task in list_of_tasks:
			taskvectors.append(get_doc_vector(model, task))
		
		for file in cleaned_files:
			newline = str(file[0])
			for index, task in enumerate(list_of_tasks):
				dist = np.linalg.norm(taskvectors[index]-get_doc_vector(model, file[4]))
				newline += " " + str(dist)
				
			f.write(newline + "\n")
			scores.append(newline.split(" "))
				
		f.close()
	return scores

def getDist(fileid, distances, task):
	#Should be possible to look up by value? but I'm lazy and performance doesn't change much
	for dist in distances:
		if fileid == dist[0]:
			return float(dist[task+1])
	else:
		#Not in top 1000 distances
		return -1

def mixDistancesScores(results, distances, task, q, mixer=1):
	distances = np.array(distances)
	resultlist = [] #will be an array with three columns: id, score, distance, mixed_score
	maxscore = -1
	minscore = 999
	maxdist = 0
	mindist = 99999

	#First append the ones we found through search
	for hit in results:
		#Find the distance of this hit to this task.
		d = getDist(hit.meta.id, distances, task)
		
	
		resultlist.append([hit.meta.id,hit.meta.score,d,-1])
		if hit.meta.score > maxscore:
			maxscore = hit.meta.score
		if hit.meta.score < minscore and hit.meta.score >= 0:
			minscore = hit.meta.score
		#print(d)
		#print(maxdist)
		if d > maxdist and d <= 1:
			maxdist = d
		if d < mindist and d >= 0:
			mindist = d

			
	#Then append the 1000 closest to this task.
	
	#First find the top 1000
	print(distances.shape)
	distances = distances[distances[:,task+1].argsort()]
	distances = distances[0:1000]
	
	
	#Then add them
	for dist in distances:
		#print(float(dist[task+1]))
		#for each file, find bm25 score to this fileid
		s = get_doc_score(q, dist[0])
		
		resultlist.append([dist[0],s,float(dist[task+1]),-1])
		
		#note: we are looking for the minimum dist..
		if float(dist[task+1]) > maxdist and float(dist[task+1]) <= 1:
			maxdist = float(dist[task+1])
		if float(dist[task+1]) < mindist and float(dist[task+1]) >= 0:
			mindist = float(dist[task+1])
		if s > maxscore:
			maxscore = s
		if s < minscore and s >= 0:
			minscore = s
			
#		print([dist[0],s,float(dist[t+2]),-1])
		
	
	
	#TODO Make it a set
	
#	print("hello")
#	print(mindist)
#	print(maxdist)
#	print(minscore)
#	print(maxscore)
	#Normalize. Done so verbose for debugging
	for index, result in enumerate(resultlist):
		#If it didn't appear in the top results (i.e. is negative) of a task, set the distance to max
		if(float(result[2]) < 0):
			resultlist[index][2] = 1
		#Otherwise normalize it 
		else:
			resultlist[index][2] = (float(resultlist[index][2]) - mindist) / (float(maxdist) - mindist)

		#If the score was not set (ie not found by query), set to minimum score possible
		if(float(result[1]) < 0):
			resultlist[index][1] = 0
		else:
			#If it was not , normalize
			resultlist[index][1] = (float(resultlist[index][1]) - minscore) / (float(maxscore) - minscore)

		#combine the scores
	#	print(resultlist[index][1])
	#	print(resultlist[index][2])
		resultlist[index][3] = resultlist[index][1] + mixer * (1 - resultlist[index][2])

			#Take the minimum because due to float computation errors we can get values over the max distance
			#TODO something is terribly wrong. I should be getting distances ranging from 0 to 1, with the lower ones scoring higher overall.
			#TODO for some reason the filter by id cannot find anything.
			
#			resultlist[index][2] = float(resultlist[index][2]) / float(maxdist)
	
#	print(maxscore)
#	print(maxdist)
	#Compute the new scores
#	for index, result in enumerate(resultlist):
		#Don't forget to normalize scores
#		print(result)
#		print(result)
#		print(maxscore)
#		print((result[1] / maxscore))
#		sleep(1)
	#	resultlist[index][1] = (result[1] / maxscore)
	#	resultlist[index][2] = (float(result[2]) / float(maxdist))
	
		#Take the average of the score and the inverted distance
#		resultlist[index][3] = result[1] + mixer * (1 - result[2])

	f=open('testme', 'w')
	for item in resultlist:
		f.write(str(item) + "\n")
	f.close()

		
	#Reorder the results
	resultlist = np.array(resultlist)
	resultlist = resultlist[resultlist[:,3].argsort()]
	
	print(resultlist)#[0:5][3]
	
	return resultlist

# Performs query, 
def reranking(results, task, model=preparedoc2vec("./covid-doc2vec.model"), mixer=1):
	taskvector = get_doc_vector(model, list_of_tasks[task])
	#print(task)

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

#NOTE the highlighter returns multiple sentencess
def printtop(results, num):
	for i in range(0, num):
		print("\n" + str(i))
		print("TITLE  " + results[i]['title'])
		if(hasattr(results[i].meta, "highlight")):
			if(hasattr(results[i].meta.highlight, "abstract")):
				print("ABSTRACT  " + results[i].meta.highlight.abstract[0])
			if(hasattr(results[i].meta.highlight, "fulltext")):
				print("FULLTEXT  " + results[i].meta.highlight.fulltext[0])
			
		
#		print(results[i].meta.highlight.abstract)
#		print(results[i].meta.highlight.fulltext)
#		for field in results[i].meta.highlight:
#			print(field)
#		if(hasattr(results[i].highlight, 'highlight')):
#			print(results[i].meta)
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


def compareSearch(results1, results2, title1="BM25", title2="fulltext", numresults = 10):
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

def countGTruth(results, gtruth):
	counter = 0
	for hit in results:
		if(hit.meta.id in gtruth):
			counter += 1
	return counter


#Starting point!

#set up elastic
indexName = "3-27-4-covid"
es_client = Elasticsearch(http_compress=True)

#Process the files: clean them and index them
processfiles("biorxiv_medrxiv")
processfiles("comm_use_subset")
processfiles("pmc_custom_license")
processfiles("noncomm_use_subset")

#Now we train or load a doc2vec model
model = preparedoc2vec("./covid-doc2vec.model")


#Let's get a ground truth for the tasks
gtruths = []
for tsk in list_of_tasks_short:
	gtruths.append(getGroundtruth(task_1_short, 100))



#So now let's see what happens if we rerank the top x documents based on distances
q = "coronavirus"
t = 1 #task
m = 1 #mixer
results_bm25 = query(q, 100)
results_reranked = reranking(results_bm25, t, model, m)

print("\n\nQuery: " + q)
print("Task: " + list_of_tasks_short[t])
#compareSearch(results_bm25, results_reranked)

print(countGTruth(results_bm25, gtruths[t]))
print(countGTruth(results_reranked, gtruths[t]))

print()
print()
print("")
# This does not alter the results a lot. Let's try making sure the top x results of each task are included in the search results

# Get distances of each doc's abstract to each task
distances = docToTaskScores("docscores")
mixedresults = mixDistancesScores(results_bm25, distances, t, q)

compareSearch(results_bm25, mixedresults)

#Another way to compare... 
#Compare word cloud of top 100 results

#wordcloud(results_bm25[0:100])
#wordcloud(results_reranked[0:100])

#plt.show()