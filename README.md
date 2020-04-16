# COVIDsearch
Playing with the research papers from the Kaggle challenge and annotations from the upcoming TREC track.

The main idea is to (re)rank and compare results based on three levels of context
1) Just the query
2) Query + fulltext search task descriptions, based on the tasks at Kaggle
3) Query + domain-specific task model

## Fulltext rerank
Create a doc2vec model based on the paper abstracts and the task descriptions. Then find out which papers are closest to which tasks.
An incoming query is automagically classified into a task, and search results are reranked based on the doc2vec distance of those docs
to the current task.

## Task-model
We did a task-analysis to create a faceted task model. For each facet a reranking function will be defined. Incoming queries will be
manually classified into this model, and we compare if this improves our results.

## What's here
covid-doc2vec.model is a gensim model trained on all paper abstracts (~april 10)
docscores has the distance of each document to each of the 10 tasks in vector space