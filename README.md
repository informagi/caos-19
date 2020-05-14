# COVIDsearch
Experimenting with using search task context to improve search results in the
TREC COVIDsearch track. Dataset is CORD-19: a collection of research papers
about the virus. See the run descriptions below. The code is slightly messy.

## What's here
| Filename | Description |
| --- | --- |
| `classifydocs.py` | the document classifying topics into kaggle tasks |
| `constants.py` | fulltext searchtask (i.e. Kaggle task) descriptions |
| `covid_search_task_expansion.py` | file for creating runs using expanded queries using search tasks (rounds 1 and 2) |
| `helpers.py` | file to help clean and index documents |
| `prep.py` | file used to prepare analysis in round 1, with some functions still used |
| `round1code.py` | the file that makes the doc2vec run (round 1) |
| `round2-run2.py` | task-based query expansion based on qrels from round1 |
| `weighted-terms.txt` | kaggle-task based query terms. tf idf weights compared to paper abstracts |
| [`anomalies.md`](./anomalies.md) | documents anomalies found in the dataset |

# Run descriptions:

## Round 2 runs

## (Manual) ru-t-exp-rnd2
Anserini bm25 (title+abstract+paragraph index) using query expansion based on Kaggle task descriptions. TREC topics were manually classified into Kaggle tasks, and the top 10 keywords of the task were extracted based on TF-IDF (using paper abstracts as a corpus).

Query terms were weighted as
a * topic_query + (1-a) * task_terms

i.e. the weights of the terms in the topic query add to 'a'
a=.8 was selected by optimizing towards both high precision on known qrels and a high number of unknown qrels.

##ru-tw-exp-rnd2
Anserini bm25 (title+abstract+paragraph index) using query expansion based on Kaggle task descriptions. TREC topics were manually classified into Kaggle tasks. All relevant documents associated with a topic in a given task was associated with that task. Based on TF-IDF (with paper abstracts as corpus) we selected n=50 keywords that characterized this task.

Words that occurred in more than 3 tasks were removed as stopwords, leaving a much smaller set of terms with tf-idf score.

Query terms were weighted as
a * topic_query + (1-a) * task_terms
a=.8 was selected because we use it in our run ruir-t-exp-rnd2. The task terms were normalized to add to 1-a based on their tf-idf scores

##ru-tn-exp-rnd2
Anserini bm25 (title+abstract+paragraph index) using query expansion based on Kaggle task descriptions. TREC topics were manually classified into Kaggle tasks, and the top 10 keywords of the task were extracted based on TF-IDF (using paper abstracts as a corpus).

Query terms were weighted as
.6 * topic_query + .25 * topic_narrative + .15 task_description
Weights were selected by trial

## Round 1 runs

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
