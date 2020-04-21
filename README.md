# COVIDsearch
Playing with the research papers from the Kaggle challenge and annotations from the upcoming TREC track.

The main idea is to (re)rank and compare results based on three levels of context
1) Just the query
2) Query + fulltext search task descriptions, based on the tasks at Kaggle
3) Query + domain-specific task model

## What's here
readtrec.py - the file that makes the runs
scores_covid19 - anserini bm25 scores
topics-rnd1 - the TREC topics for round 1

prep.py - reads in/cleans the files and indexes them. I used elastic for convenience - should change to Anserini  
helpers.py - helps prep.py  
constants.py - fulltext searchtask (i.e. Kaggle task) descriptions  
analysis.py - word clouds, clustering, visualisation - used to explore data and see if ranking makes sense  
search.py - the (re)ranking and searching

docscores - has the distance of each document to each of the 10 tasks in vector space  
covid-doc2vec.model - gensim doc2vec model trained on all paper abstracts (april 10 version)

## Fulltext rerank
Create a doc2vec model based on the paper abstracts and the task descriptions. Then find out which papers are closest to which tasks.
An incoming query is automagically classified into a task, and search results are reranked based on the doc2vec distance of those docs
to the current task.

## Task-model
We did a task-analysis to create a faceted task model. For each facet a reranking function will be defined. Incoming queries will be
manually classified into this model, and we compare if this improves our results.
