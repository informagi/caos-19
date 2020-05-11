from constants import *
from rank_bm25 import BM25Okapi

'''
Let's try to classify each topic by making an index of the tasks
and ranking them for each query
'''

manual_classification = [2, 0, 3, 0, 3, 7, 3, 7, 6, 5, 4, 5, 0, 0, 0, 0, 7, 5, 5, 1, 0, 0, 1, 1, 1, 7, 7, 3, 3, 3]

tokenized_corpus = [doc.split(" ") for doc in list_of_tasks]
bm25 = BM25Okapi(tokenized_corpus)

query = "coronavirus in Canada"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)
