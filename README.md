# COVIDsearch
Experimenting with using search task context to improve search results in the
TREC COVIDsearch track. Dataset is CORD-19: a collection of research papers
about the virus. See the run descriptions below. The code is slightly messy.

## What's here
| Filename | Description |
| --- | --- |
| `readtrec.py` | the file that makes the doc2vec run (Chris has code for the others) |
| `classifydocs.py` | the document classifying |
| `constants.py` | fulltext searchtask (i.e. Kaggle task) descriptions |
| `prep.py` | reads in/cleans the files and indexes them. I used elastic for convenience - should change to Anserini |
| `helpers.py` | helps `prep.py` |
| `analysis.py` | word clouds, clustering, visualisation - used to explore data and see if ranking makes sense |
| `search.py` | where I tried out some stuff with (re)ranking and searching |
| `docscores` | has the distance of each document to each of the 10 tasks in vector space |
| `covid-doc2vec.model` | gensim doc2vec model trained on all paper abstracts (2020-04-10 version) |
| `anserini_bm25.txt` | anserini bm25 scores |
| `topics-rnd1` | the TREC topics for round 1 |
| [`anomalies.md`][1] | documents anomalies found in the dataset |

# Run descriptions:

## (Manual) run RUIR-doc2vec
We interpreted the [Kaggle tasks][0] as descriptions of search tasks, and
boosted documents relevant to the given search task.

TREC topics were manually classified into the most appropriate
searchtask/kaggle task (in topic order:
[2,0,3,0,3,7,3,7,6,5,4,5,0,0,0,0,7,5,5,1,0,0,1,1,1,7,7,3,3,3], referring to
tasks in the order of the Kaggle tasks page accessed 2020-04-23).

To find out which documents are relevant to which tasks, we trained a doc2vec
model on the paper abstracts and fulltext task descriptions on Kaggle. We
retrieved the top 1000 results using Anserini bm25 (fulltext+title+abstract),
and reranked them based on the distance between a task description and paper
abstract in doc2vec space. BM25 scores of the top 1000 documents were
normalized to range from 0 to 1. The same happened for the distances between
the paper abstracts and task descriptions. These two scores were then added.

## (Manual) run RUIR-bm25-mt-exp

We interpreted the [Kaggle tasks][0] as descriptions of search tasks, and
performed query expansion using keywords typical for a given search task.

TREC topics were manually classified into the most appropriate
searchtask/kaggle task (in topic order:
[2,0,3,0,3,7,3,7,6,5,4,5,0,0,0,0,7,5,5,1,0,0,1,1,1,7,7,3,3,3], referring to
tasks in the order of the Kaggle tasks page accessed 2020-04-23).

The keywords for expansion were automatically selected from the fulltext task
descriptions on Kaggle. The words in that text were ranked by TF-IDF, and words
that clearly are not about the topic were filtered (~1 per task, except for the
sample task). Then the top 10 words were selected.

These keywords were used as expansion terms to enrich the topic query of the
corresponding task. This was done by appending these keywords to the query.
Using this enriched query we ranked the documents using Anserini bm25
(fulltext+title+abstract).

## (Automatic) run RUIR-bm25-at-exp
We interpreted the [Kaggle tasks][0] as descriptions of search tasks, and
performed query expansion using keywords typical for a given search task.

TREC topics were automatically classified into the most appropriate
searchtask/kaggle task by indexing the fulltext of the tasks and ranking them
based on the topic query. The top result was selected as the classification.

The keywords for expansion were automatically selected from the fulltext task
descriptions on Kaggle. The words in that text were ranked by TF-IDF, and words
that clearly are not about the topic were filtered (~1 per task, except for the
sample task). Then the top 10 words were selected.

The keywords were curated from the fulltext task descriptions on Kaggle
(between 10 to 22 keywords were identified for each task). Then these keywords
were used as expansion terms to enrich the query described in the topic by
appending these keywords to the query. Using this enriched query we ranked the
documents using Anserini bm25 (fulltext+title+abstract).

[0]: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks
[1]: ../blob/master/anomalies.md
