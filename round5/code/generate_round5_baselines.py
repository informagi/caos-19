#
# Pyserini: Python interface to the Anserini IR toolkit built on Lucene
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Perform Anserini baseline runs for TREC-COVID Round 4."""

import hashlib
import os
import sys

from covid_baseline_tools import evaluate_runs, verify_stored_runs

sys.path.insert(0, './')
sys.path.insert(0, '../pyserini/')

from pyserini.util import compute_md5


# This makes errors more readable,
# see https://stackoverflow.com/questions/27674602/hide-traceback-unless-a-debug-flag-is-set
sys.tracebacklimit = 0

indexes = ['C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/indexes/lucene-index-cord19-abstract-2020-07-16',
'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/indexes/lucene-index-cord19-full-text-2020-07-16',
'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/indexes/lucene-index-cord19-paragraph-2020-07-16']
"""
cumulative_runs = {
    'anserini.covid-r5.abstract.qq.bm25.txt': 'b1ccc364cc9dab03b383b71a51d3c6cb',
    'anserini.covid-r5.abstract.qdel.bm25.txt': 'ee4e3e6cf87dba2fd021fbb89bd07a89',
    'anserini.covid-r5.full-text.qq.bm25.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.qdel.bm25.txt': '8387e4ad480ec4be7961c17d2ea326a1',
    'anserini.covid-r5.paragraph.qq.bm25.txt': '62d713a1ed6a8bf25c1454c66182b573',
    'anserini.covid-r5.paragraph.qdel.bm25.txt':  '16b295fda9d1eccd4e1fa4c147657872',
    'anserini.covid-r5.fusion1.txt': '16875b6d32a9b5ef96d7b59315b101a7',
    'anserini.covid-r5.fusion2.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.abstract.ruir1.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.abstract.ruir2.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.abstract.ruir3.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.paragraph.ruir1.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.paragraph.ruir2.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.paragraph.ruir3.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.full-text.ruir1.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.full-text.ruir2.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.full-text.ruir3.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'ruir.fusion1.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'ruir.fusion2.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'ruir.fusion3.txt': '8f7d663d551f831c65dceb8e4e9219c2',
    'anserini.covid-r5.abstract.qdel.bm25+rm3Rf.txt': '909ccbbd55736eff60c7dbeff1404c94',
    'anserini.covid-r5.full-text.qq.bm25.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.qdel.bm25.txt': '8387e4ad480ec4be7961c17d2ea326a1',
    'anserini.covid-r5.full-text.ruir31.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruir32.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruir33.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruir51.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruir52.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruir53.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruirm1.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruirm2.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruirm3.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruirs1.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruirs2.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.ruirs3.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruir53f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruirm3f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruirs3f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruir33f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.fusionq.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.qonly.bm25.txt': 'd7457dd746533326f2bf8e85834ecf5c',
}"""


"""
cumulative_runs = {
    'anserini.covid-r5.full-text.qonly.bm25.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.abstract.qonly.bm25.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.paragraph.qonly.bm25.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.full-text.qq.bm25.txt': 'b1ccc364cc9dab03b383b71a51d3c6cb',
    'anserini.covid-r5.abstract.qq.bm25.txt': 'b1ccc364cc9dab03b383b71a51d3c6cb',
    'anserini.covid-r5.paragraph.qq.bm25.txt': 'b1ccc364cc9dab03b383b71a51d3c6cb',
    'anserini.covid-r5.full-text.33.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.abstract.33.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.paragraph.33.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruir53f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruirm3f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruirs3f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruir33f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.fusionq.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruir33ff.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruir53ff.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.fusion1.txt': '16875b6d32a9b5ef96d7b59315b101a7',
    'anserini.covid-r5.fusion2.txt': '16875b6d32a9b5ef96d7b59315b101a7',
    'udel33f.txt': '16875b6d32a9b5ef96d7b59315b101a7',
}

"""

cumulative_runs = {
	'jp-0.0.run' : 'a',
	'jp-1.0.run' : 'a',
	'jp-2.0.run' : 'a',
	'jt-0.0.run' : 'a',
	'jt-1.0.run' : 'a',
	'jt-2.0.run' : 'a',
#	'1-RUIR-doc2vec.txt': 'nah',
#	'2-anserini_bm25.txt': 'han',
	'anserini.covid-r5.full-text.qonly.bm25.txt' : 'test',
	'anserini.covid-r5.abstract.qonly.bm25.txt' : 'test',
	'anserini.covid-r5.paragraph.qonly.bm25.txt' : 'test',
    'anserini.covid-r5.full-text.33.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.abstract.33.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.paragraph.33.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.fusionq.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'anserini.covid-r5.fusion1.txt': '16875b6d32a9b5ef96d7b59315b101a7',
    'anserini.covid-r5.fusion2.txt': '16875b6d32a9b5ef96d7b59315b101a7',
    'ruir33f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'ruir33ff.txt': 'd7457dd746533326f2bf8e85834ecf5c',
    'udel33f.txt': '16875b6d32a9b5ef96d7b59315b101a7',
    'ruir53f.txt': 'd7457dd746533326f2bf8e85834ecf5c',
}

final_runs = {
    'anserini.final-r5.fusion1.txt': '12122c12089c2b07a8f6c7247aebe2f6',
    'anserini.final-r5.fusion2.txt': 'ff1a0bac315de6703b937c552b351e2a',
    'anserini.final-r5.rf.txt': '74e2a73b5ffd2908dc23b14c765171a1',
    'final.ruir1.txt':'74e2a73b5ffd2908dc23b14c765171a1',
    'final.ruir2.txt':'74e2a73b5ffd2908dc23b14c765171a1',
    'final.ruir3.txt':'74e2a73b5ffd2908dc23b14c765171a1',
    'final.ruir33.txt':'74e2a73b5ffd2908dc23b14c765171a1',
    'final.ruir53.txt':'74e2a73b5ffd2908dc23b14c765171a1',
    'final.ruirm3.txt':'74e2a73b5ffd2908dc23b14c765171a1',
    'final.ruirs3.txt':'74e2a73b5ffd2908dc23b14c765171a1'
}

"""
stored_runs = {
    'https://www.dropbox.com/s/lbgevu4wiztd9e4/anserini.covid-r5.abstract.qq.bm25.txt?dl=1':
        cumulative_runs['anserini.covid-r5.abstract.qq.bm25.txt'],
    'https://www.dropbox.com/s/pdy5o4xyalcnm2n/anserini.covid-r5.abstract.qdel.bm25.txt?dl=1':
        cumulative_runs['anserini.covid-r5.abstract.qdel.bm25.txt'],
    'https://www.dropbox.com/s/zhrkqvgbh6mwjdc/anserini.covid-r5.full-text.qq.bm25.txt?dl=1':
        cumulative_runs['anserini.covid-r5.full-text.qq.bm25.txt'],
    'https://www.dropbox.com/s/4c3ifc8gt96qiio/anserini.covid-r5.full-text.qdel.bm25.txt?dl=1':
        cumulative_runs['anserini.covid-r5.full-text.qdel.bm25.txt'],
    'https://www.dropbox.com/s/xfx3g54map005sy/anserini.covid-r5.paragraph.qq.bm25.txt?dl=1':
        cumulative_runs['anserini.covid-r5.paragraph.qq.bm25.txt'],
    'https://www.dropbox.com/s/nmb11wtx4yde939/anserini.covid-r5.paragraph.qdel.bm25.txt?dl=1':
        cumulative_runs['anserini.covid-r5.paragraph.qdel.bm25.txt'],
    'https://www.dropbox.com/s/mq94s9t7snqlizw/anserini.covid-r5.fusion1.txt?dl=1':
        cumulative_runs['anserini.covid-r5.fusion1.txt'],
    'https://www.dropbox.com/s/4za9i29gxv090ut/anserini.covid-r5.fusion2.txt?dl=1':
        cumulative_runs['anserini.covid-r5.fusion2.txt'],
    'https://www.dropbox.com/s/9cw0qhr5meskg9y/anserini.covid-r5.abstract.qdel.bm25%2Brm3Rf.txt?dl=1':
        cumulative_runs['anserini.covid-r5.abstract.qdel.bm25+rm3Rf.txt'],
    'https://www.dropbox.com/s/2uyws7fnbpxo8s6/anserini.final-r5.fusion1.txt?dl=1':
        final_runs['anserini.final-r5.fusion1.txt'],
    'https://www.dropbox.com/s/vyolaecpxu28vjw/anserini.final-r5.fusion2.txt?dl=1':
        final_runs['anserini.final-r5.fusion2.txt'],
    'https://www.dropbox.com/s/27wy54cibmyg7lp/anserini.final-r5.rf.txt?dl=1':
        final_runs['anserini.final-r5.rf.txt']
}"""

def perform_runs(cumulative_qrels):
    base_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5.xml'
    udel_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-udel.xml'
    ruir1_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruir.qdel.weight1.tfidf3.xml' #udel w1
    ruir2_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruir.qdel.weight2.tfidf3.xml' #udel w2
    ruir3_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruir.qq.weight1.tfidf3.xml'#qq w1
    ruir31_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruir31.xml'
    ruir32_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruir32.xml'
    ruir33_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruir33.xml'

    ruir51_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruir51.xml'
    ruir52_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruir52.xml'
    ruir53_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruir53.xml'

    ruirm1_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruirm1.xml'
    ruirm2_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruirm2.xml'
    ruirm3_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruirm3.xml'

    ruirs1_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruirs1.xml'
    ruirs2_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruirs2.xml'
    ruirs3_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-ruirs3.xml'

    udel33_topics = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/topics.covid-round5-udel33.xml'
    
    print('')
    print('## Running on abstract index...')
    print('')

    abstract_index = indexes[0]
    abstract_prefix = 'anserini.covid-r5.abstract'


    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {abstract_index} ' +
              f'-topicreader Covid -topics {udel33_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{abstract_prefix}.udel33.txt -runtag {abstract_prefix}.udel33.txt')


    
    """
    

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {abstract_index} ' +
              f'-topicreader Covid -topics {ruir53_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{abstract_prefix}.53.txt -runtag {abstract_prefix}.53.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {abstract_index} ' +
              f'-topicreader Covid -topics {base_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{abstract_prefix}.qonly.bm25.txt -runtag {abstract_prefix}.qonly.bm25.txt')
    

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {abstract_index} ' +
              f'-topicreader Covid -topics {base_topics} -topicfield query+question ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{abstract_prefix}.qq.bm25.txt -runtag {abstract_prefix}.qq.bm25.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {abstract_index} ' +
              f'-topicreader Covid -topics {udel_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{abstract_prefix}.qdel.bm25.txt -runtag {abstract_prefix}.qdel.bm25.txt')

    

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {abstract_index} ' +
              f'-topicreader Covid -topics {udel_topics} -topicfield query -removedups ' +
              f'-bm25 -rm3 -rm3.fbTerms 100 -hits 5000 ' +
              f'-rf.qrels {cumulative_qrels} ' +
              f'-output runs/{abstract_prefix}.qdel.bm25+rm3Rf.txt -runtag {abstract_prefix}.qdel.bm25+rm3Rf.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {abstract_index} ' +
              f'-topicreader Covid -topics {ruir1_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{abstract_prefix}.ruir1.txt -runtag {abstract_prefix}.ruir1.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {abstract_index} ' +
              f'-topicreader Covid -topics {ruir2_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{abstract_prefix}.ruir2.txt -runtag {abstract_prefix}.ruir2.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {abstract_index} ' +
              f'-topicreader Covid -topics {ruir3_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{abstract_prefix}.ruir3.txt -runtag {abstract_prefix}.ruir3.txt')
    """




    print('')
    print('## Running on full-text index...')
    print('')

    full_text_index = indexes[1]
    full_text_prefix = 'anserini.covid-r5.full-text'


    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {udel33_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.udel33.txt -runtag {full_text_prefix}.udel33.txt')

    """

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir53_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir53.txt -runtag {full_text_prefix}.ruir53.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir33_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.33.txt -runtag {full_text_prefix}.33.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir53_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.53.txt -runtag {full_text_prefix}.53.txt')


    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {base_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.qonly.bm25.txt -runtag {full_text_prefix}.qonly.bm25.txt')


    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {base_topics} -topicfield query+question ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.qq.bm25.txt -runtag {full_text_prefix}.qq.bm25.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {udel_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.qdel.bm25.txt -runtag {full_text_prefix}.qdel.bm25.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir1_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir1.txt -runtag {full_text_prefix}.ruir1.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir2_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir2.txt -runtag {full_text_prefix}.ruir2.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir3_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir3.txt -runtag {full_text_prefix}.ruir3.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir31_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir31.txt -runtag {full_text_prefix}.ruir31.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir32_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir32.txt -runtag {full_text_prefix}.ruir32.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir33_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir33.txt -runtag {full_text_prefix}.ruir33.txt')


    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir51_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir51.txt -runtag {full_text_prefix}.ruir51.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir52_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir52.txt -runtag {full_text_prefix}.ruir52.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruir53_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruir53.txt -runtag {full_text_prefix}.ruir53.txt')


    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruirm1_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruirm1.txt -runtag {full_text_prefix}.ruirm1.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruirm2_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruirm2.txt -runtag {full_text_prefix}.ruirm2.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruirm3_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruirm3.txt -runtag {full_text_prefix}.ruirm3.txt')


    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruirs1_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruirs1.txt -runtag {full_text_prefix}.ruirs1.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruirs2_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruirs2.txt -runtag {full_text_prefix}.ruirs2.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {full_text_index} ' +
              f'-topicreader Covid -topics {ruirs3_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{full_text_prefix}.ruirs3.txt -runtag {full_text_prefix}.ruirs3.txt')

    """    


    print('')
    print('## Running on paragraph index...')
    print('')

    paragraph_index = indexes[2]
    paragraph_prefix = 'anserini.covid-r5.paragraph'


    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {paragraph_index} ' +
              f'-topicreader Covid -topics {udel33_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{paragraph_prefix}.udel33.txt -runtag {paragraph_prefix}.udel33.txt')


    """
    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {paragraph_index} ' +
              f'-topicreader Covid -topics {ruir33_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{paragraph_prefix}.33.txt -runtag {paragraph_prefix}.33.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {paragraph_index} ' +
              f'-topicreader Covid -topics {ruir53_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{paragraph_prefix}.53.txt -runtag {paragraph_prefix}.53.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {paragraph_index} ' +
              f'-topicreader Covid -topics {base_topics} -topicfield query ' +
              f'-removedups -strip_segment_id -bm25 -hits 5000 ' +
              f'-output runs/{paragraph_prefix}.qonly.bm25.txt -runtag {paragraph_prefix}.qonly.bm25.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {paragraph_index} ' +
              f'-topicreader Covid -topics {base_topics} -topicfield query+question ' +
              f'-removedups -strip_segment_id -bm25 -hits 5000 ' +
              f'-output runs/{paragraph_prefix}.qq.bm25.txt -runtag {paragraph_prefix}.qq.bm25.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {paragraph_index} ' +
              f'-topicreader Covid -topics {udel_topics} -topicfield query ' +
              f'-removedups -strip_segment_id -bm25 -hits 5000 ' +
              f'-output runs/{paragraph_prefix}.qdel.bm25.txt -runtag {paragraph_prefix}.qdel.bm25.txt')
    

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {paragraph_index} ' +
              f'-topicreader Covid -topics {ruir1_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{paragraph_prefix}.ruir1.txt -runtag {paragraph_prefix}.ruir1.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {paragraph_index} ' +
              f'-topicreader Covid -topics {ruir2_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{paragraph_prefix}.ruir2.txt -runtag {paragraph_prefix}.ruir2.txt')

    os.system(f'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/target/appassembler/bin/SearchCollection -index {paragraph_index} ' +
              f'-topicreader Covid -topics {ruir3_topics} -topicfield query ' +
              f'-removedups -bm25 -hits 5000 ' +
              f'-output runs/{paragraph_prefix}.ruir3.txt -runtag {paragraph_prefix}.ruir3.txt')
    """

def perform_fusion(check_md5=True):
    check_md5=False
    print('')
    print('## Performing fusion...')
    print('')


    fusion_run33 = 'udel33f.txt'
    set5 = ['anserini.covid-r5.abstract.qdel.bm25.txt',
            'anserini.covid-r5.full-text.udel33.txt',
            'anserini.covid-r5.paragraph.qdel.bm25.txt']

    print(f'Performing fusion to create {fusion_run33}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 '
              f'--out runs/{fusion_run33} --runs runs/{set5[0]} runs/{set5[1]} runs/{set5[2]}')

    fusion_run33 = 'ruir33ff.txt'
    set5 = ['anserini.covid-r5.abstract.33.txt',
            'anserini.covid-r5.full-text.33.txt',
            'anserini.covid-r5.paragraph.qonly.bm25.txt']

    print(f'Performing fusion to create {fusion_run33}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 '
              f'--out runs/{fusion_run33} --runs runs/{set5[0]} runs/{set5[1]} runs/{set5[2]}')

    """



    fusion_run53 = 'ruir53ff.txt'
    set5 = ['anserini.covid-r5.abstract.qonly.bm25.txt',
            'anserini.covid-r5.full-text.53.txt',
            'anserini.covid-r5.paragraph.qonly.bm25.txt']

    print(f'Performing fusion to create {fusion_run53}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 '
              f'--out runs/{fusion_run53} --runs runs/{set5[0]} runs/{set5[1]} runs/{set5[2]}')

    fusion_runq = 'anserini.covid-r5.fusionq.txt'
    set1 = ['anserini.covid-r5.abstract.qonly.bm25.txt',
            'anserini.covid-r5.full-text.qonly.bm25.txt',
            'anserini.covid-r5.paragraph.qonly.bm25.txt']

    print(f'Performing fusion to create {fusion_runq}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 '
              f'--out runs/{fusion_runq} --runs runs/{set1[0]} runs/{set1[1]} runs/{set1[2]}')

    fusion_run1 = 'anserini.covid-r5.fusion1.txt'
    set1 = ['anserini.covid-r5.abstract.qq.bm25.txt',
            'anserini.covid-r5.full-text.qq.bm25.txt',
            'anserini.covid-r5.paragraph.qq.bm25.txt']

    print(f'Performing fusion to create {fusion_run1}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 '
              f'--out runs/{fusion_run1} --runs runs/{set1[0]} runs/{set1[1]} runs/{set1[2]}')


    fusion_run33 = 'ruir33f.txt'
    set5 = ['anserini.covid-r5.abstract.qonly.bm25.txt',
            'anserini.covid-r5.full-text.ruir33.txt',
            'anserini.covid-r5.paragraph.qonly.bm25.txt']

    print(f'Performing fusion to create {fusion_run33}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 ' +
              f'--out runs/{fusion_run33} --runs runs/{set5[0]} runs/{set5[1]} runs/{set5[2]}')


    fusion_run1 = 'anserini.covid-r5.fusion1.txt'
    set1 = ['anserini.covid-r5.abstract.qq.bm25.txt',
            'anserini.covid-r5.full-text.qq.bm25.txt',
            'anserini.covid-r5.paragraph.qq.bm25.txt']

    print(f'Performing fusion to create {fusion_run1}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 '
              f'--out runs/{fusion_run1} --runs runs/{set1[0]} runs/{set1[1]} runs/{set1[2]}')

    if check_md5:
        assert compute_md5(f'runs/{fusion_run1}') == cumulative_runs[fusion_run1], f'Error in producing {fusion_run1}!'

    fusion_run2 = 'anserini.covid-r5.fusion2.txt'
    set2 = ['anserini.covid-r5.abstract.qdel.bm25.txt',
            'anserini.covid-r5.full-text.qdel.bm25.txt',
            'anserini.covid-r5.paragraph.qdel.bm25.txt']

    print(f'Performing fusion to create {fusion_run2}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 ' +
              f'--out runs/{fusion_run2} --runs runs/{set2[0]} runs/{set2[1]} runs/{set2[2]}')

    if check_md5:
        assert compute_md5(f'runs/{fusion_run2}') == cumulative_runs[fusion_run2], f'Error in producing {fusion_run2}!'




    fusion_run_r1 = 'ruir.fusion1.txt'
    set3 = ['anserini.covid-r5.abstract.ruir1.txt',
            'anserini.covid-r5.full-text.ruir1.txt',
            'anserini.covid-r5.paragraph.ruir1.txt']

    print(f'Performing fusion to create {fusion_run_r1}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 ' +
              f'--out runs/{fusion_run_r1} --runs runs/{set3[0]} runs/{set3[1]} runs/{set3[2]}')

    fusion_run_r2 = 'ruir.fusion2.txt'
    set4 = ['anserini.covid-r5.abstract.ruir2.txt',
            'anserini.covid-r5.full-text.ruir2.txt',
            'anserini.covid-r5.paragraph.ruir2.txt']

    print(f'Performing fusion to create {fusion_run_r2}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 ' +
              f'--out runs/{fusion_run_r2} --runs runs/{set4[0]} runs/{set4[1]} runs/{set4[2]}')

    fusion_run_r3 = 'ruir.fusion3.txt'
    set5 = ['anserini.covid-r5.abstract.ruir3.txt',
            'anserini.covid-r5.full-text.ruir3.txt',
            'anserini.covid-r5.paragraph.ruir3.txt']

    print(f'Performing fusion to create {fusion_run_r3}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 ' +
              f'--out runs/{fusion_run_r3} --runs runs/{set5[0]} runs/{set5[1]} runs/{set5[2]}')



    fusion_run53 = 'ruir53f.txt'
    set5 = ['anserini.covid-r5.abstract.qq.bm25.txt',
            'anserini.covid-r5.full-text.ruir53.txt',
            'anserini.covid-r5.paragraph.qq.bm25.txt']

    print(f'Performing fusion to create {fusion_run53}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 ' +
              f'--out runs/{fusion_run53} --runs runs/{set5[0]} runs/{set5[1]} runs/{set5[2]}')


    fusion_runm3 = 'ruirm3f.txt'
    set5 = ['anserini.covid-r5.abstract.qq.bm25.txt',
            'anserini.covid-r5.full-text.ruirm3.txt',
            'anserini.covid-r5.paragraph.qq.bm25.txt']

    print(f'Performing fusion to create {fusion_runm3}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 ' +
              f'--out runs/{fusion_runm3} --runs runs/{set5[0]} runs/{set5[1]} runs/{set5[2]}')

    fusion_runs3 = 'ruirs3f.txt'
    set5 = ['anserini.covid-r5.abstract.qq.bm25.txt',
            'anserini.covid-r5.full-text.ruirs3.txt',
            'anserini.covid-r5.paragraph.qq.bm25.txt']

    print(f'Performing fusion to create {fusion_runs3}')
    os.system('PYTHONPATH=../pyserini ' +
              'python -m pyserini.fusion --method rrf --runtag reciprocal_rank_fusion_k=60 --k 10000 ' +
              f'--out runs/{fusion_runs3} --runs runs/{set5[0]} runs/{set5[1]} runs/{set5[2]}')

    """


def prepare_final_submissions(cumulative_qrels, check_md5=False):
    print('')
    print('## Preparing final submission files by removing qrels...')
    print('')

    run1 = 'anserini.final-r5.fusion1.txt'
    print(f'Generating {run1}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/anserini.covid-r5.fusion1.txt --output runs/{run1} --runtag r5.fusion1')
    run1_md5 = compute_md5(f'runs/{run1}')
    if check_md5:
        assert run1_md5 == final_runs[run1], f'Error in producing {run1}!'

    run2 = 'anserini.final-r5.fusion2.txt'
    print(f'Generating {run2}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/anserini.covid-r5.fusion2.txt --output runs/{run2} --runtag r5.fusion2')
    run2_md5 = compute_md5(f'runs/{run2}')
    if check_md5:
        assert run2_md5 == final_runs[run2], f'Error in producing {run2}!'

    run3 = 'anserini.final-r5.rf.txt'
    print(f'Generating {run3}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/anserini.covid-r5.abstract.qdel.bm25+rm3Rf.txt --output runs/{run3} --runtag r5.rf')
    run3_md5 = compute_md5(f'runs/{run3}')
    if check_md5:
        assert run3_md5 == final_runs[run3], f'Error in producing {run3}!'


    run4 = 'final.ruir1.txt'
    print(f'Generating {run4}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/ruir.fusion1.txt --output runs/{run4} --runtag r5.rf')
    run4_md5 = compute_md5(f'runs/{run4}')

    run5 = 'final.ruir2.txt'
    print(f'Generating {run5}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/ruir.fusion2.txt --output runs/{run5} --runtag r5.rf')
    run5_md5 = compute_md5(f'runs/{run5}')

    run6 = 'final.ruir3.txt'
    print(f'Generating {run6}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/ruir.fusion3.txt --output runs/{run6} --runtag r5.rf')
    run6_md5 = compute_md5(f'runs/{run6}')


    run33 = 'final.qruir33.txt'
    print(f'Generating {run33}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/ruir33f.txt --output runs/{run33} --runtag final.qruir33.txt')
    run33_md5 = compute_md5(f'runs/{run33}')

    run52 = 'final.ruir52.txt'
    print(f'Generating {run52}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/ruir52f.txt --output runs/{run52} --runtag r5.rf')
    run52_md5 = compute_md5(f'runs/{run52}')

    runm2 = 'final.ruirm2.txt'
    print(f'Generating {runm2}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/ruirm2f.txt --output runs/{runm2} --runtag r5.rf')
    runm2_md5 = compute_md5(f'runs/{runm2}')

    runs2 = 'final.ruirs2.txt'
    print(f'Generating {runs2}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/ruirs2f.txt --output runs/{runs2} --runtag r5.rf')
    runs2_md5 = compute_md5(f'runs/{runs2}')

    runq = 'final.qruir.txt'
    print(f'Generating {runq}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/anserini.covid-r5.fusionq.txt --output runs/{runq} --runtag final.qruir.txt')
    runs2_md5 = compute_md5(f'runs/{runq}')

    runq = 'final.qruir.filtered.txt'
    print(f'Generating {runq}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/ruir33f.txt-filtered --output runs/{runq} --runtag final.qruir.filtered.txt')
    runs2_md5 = compute_md5(f'runs/{runq}')



    runq = 'final.qonly.txt'
    print(f'Generating {runq}')
    os.system(f'python tools/scripts/filter_run_with_qrels.py --discard --qrels {cumulative_qrels} ' +
              f'--input runs/anserini.covid-r5.full-text.qonly.bm25.txt --output runs/{runq} --runtag final.qonly.txt')
    runsq_md5 = compute_md5(f'runs/{runq}')





    print('')
    print(run1 + ' ' * (35 - len(run1)) + run1_md5)
    print(run2 + ' ' * (35 - len(run2)) + run2_md5)
    print(run3 + ' ' * (35 - len(run3)) + run3_md5)
    print(run4 + ' ' * (35 - len(run4)) + run4_md5)
    print(run5 + ' ' * (35 - len(run5)) + run5_md5)
    print(run6 + ' ' * (35 - len(run6)) + run6_md5)
    print(run33 + ' ' * (35 - len(run33)) + run33_md5)
    print(run52 + ' ' * (35 - len(run52)) + run52_md5)
    print(runm2 + ' ' * (35 - len(runm2)) + runm2_md5)
    print(runs2 + ' ' * (35 - len(runs2)) + runs2_md5)


def main():
    if not (os.path.isdir(indexes[0]) and os.path.isdir(indexes[1]) and os.path.isdir(indexes[2])):
        print('Required indexes do not exist. Please download first.')

    cumulative_qrels = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/qrels.covid-round12.txt'
    cumulative_qrels = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/qrels.covid-round4-30topics.txt'
    cumulative_qrels = 'C:/Users/Allemaal/Desktop/ubuntu/Desktop/anserini/src/main/resources/topics-and-qrels/qrels.covid-round4-cumulative.txt'

    #verify_stored_runs(stored_runs)
    #perform_runs(cumulative_qrels)
    #perform_fusion(check_md5=False)
    #prepare_final_submissions(cumulative_qrels, check_md5=False)
	
	
	
    evaluate_runs(cumulative_qrels, cumulative_runs, check_md5=False)


if __name__ == '__main__':
    main()
