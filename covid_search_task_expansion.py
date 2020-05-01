import xml.etree.ElementTree as ET

from pyserini.search import pysearch

tree = ET.parse('topics-rnd1.xml')
root = tree.getroot()

tasks = [
    ['environmental transmission', 'incubation', 'contagious', 'persistence', 'stability', 'physical', 'weather', 'epidemiology', 'shedding', 'reproductive number', 'modes of transmission', 'virulent', 'asymptomatic', 'pathogen', 'evolutionary host', 'transmission host'],
    ['smoking', 'risk', 'pulmonary', 'pre-condition', 'co-infection', 'high-risk', 'severe', 'susceptible', 'fatality', 'neonates', 'respitory', 'condition', 'pre-existing', 'pregnant', 'morbidities'],
    ['human-animal', 'origin', 'genetics', 'evolution', 'genome', 'sample sets', 'genomic', 'strain', 'livestock', 'animal host', 'natural history', 'genetic drift', 'mutation', 'genomics', 'sequencing'],
    ['vaccine','therapeutic','treat','drugs','pharmaceuticals','recipients','ADE','complication','antiviral','prophylaxis','cloroquine','vaccination','immume respone'],
    ['medical care','surge capacity','nursing home','allocation','personal protective equirement','clinical characterization','nursing','care','Extracorporeal membrane oxygenation','ECMO','mechanical ventilation','extrapulmonary manifestations','cardiomyopathy','cardiac arrest','regulatory standards','N95 masks','elastomeric respirators','telemedicine','steroids','high flow oxygen','supportive interventions'],
    ['NPI','non-pharmaceutical intervention','school closure','travel ban','quarantine','mass gathering','social distancing','public health advice','economic impact'],
    ['counties','geographic','geography','mortality rate','spread','mutations'],
    ['diagnostic','surveillance','detection','screening','ELISAs','capacity','testing','point-of-care','rapid testing','pathogen','reagent','cytokines','response markers','swabs'],
    ['ethical','social science','principles','standards','ethics','psychological health','fear','anxiety','stigma','sociology'],
    ['collaboration','nomenclature','data standards','information sharing','communication','collaborate','coordination','misunderstanding','action plan']
]


topic_tasknr = [2,0,3,0,3,7,3,7,6,5,4,5,0,0,0,0,7,5,5,1,0,0,1,1,1,7,7,3,3,3]

from tqdm import tqdm

with open('RU-bm25-t-exp-1000.txt', 'w') as f:
    for i in tqdm(range(30)):
        query = root[i][0].text
        task = topic_tasknr[i]
        expansion = tasks[task]
        query = query.split(' ')
        for w in expansion:
            for part in w.split(' '):
                query.append(part)
        query = ' '.join(query)
        searcher = pysearch.SimpleSearcher('lucene-index-covid-full-text-2020-04-10/')
        hits = searcher.search(query, 1005)
        topicno = i + 1
        seen = set()
        j = 0
        while j < 1200:
            if hits[j].docid in seen:
                j+=1
                continue
            rank = len(seen) + 1
            hit = hits[j]
            f.write(f'{topicno} Q0 {hit.docid} {rank} {hit.score} RU-bm25-t-exp\n')
            seen |= {hits[j].docid}
            if rank == 1000:
                break
            j+=1

