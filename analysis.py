from prep import *

#create a list of the taskvectors
taskvectors = []
for task in list_of_tasks:
	taskvectors.append(get_doc_vector(model, task))
	
def getGroundtruth(q, size=1000):
	results = query(q, size)
	ids = []
	for result in results:
#		print(result.meta.id)
		ids.append(result.meta.id)
	return ids