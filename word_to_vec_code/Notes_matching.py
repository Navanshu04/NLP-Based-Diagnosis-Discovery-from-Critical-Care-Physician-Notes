import math
import re
from collections import defaultdict

import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial import distance

from sentence2vec import Word, Sentence, sentence_to_vec


# euclidean distance between two vectors
def l2_dist(v1, v2):
    sum = 0.0
    if len(v1) == len(v2):
        for i in range(len(v1)):
            delta = v1[i] - v2[i]
            sum += delta * delta
        return math.sqrt(sum)


vectors = KeyedVectors.load_word2vec_format(
    "C:/Users/Asus/PycharmProjects/Mimic_3/word_to_vec_code/bio_embedding_intrinsic", binary=True)

embedding_size = 200
sentences = []
notes = pd.read_csv("C:/Users/Asus/Desktop/notes_physician_texttrim_np.csv")
notes.rename(columns={'TEXT_TRIM_NOUN_PHRASES': 'NOTE'}, inplace=True)
notes['NOTE'] = notes['NOTE'].apply(lambda x: re.sub(r'[,\[\]#\'"]', '', x))
notes['NOTE'] = notes['NOTE'].apply(lambda x: re.sub(' +', ' ', x))
notes['NOTE'] = notes['NOTE'].apply(lambda x: re.sub(r'[-/()+*:><]', ' ', x))
notes['NOTE'] = notes['NOTE'].apply(lambda x: x.lower())
sentences = list(notes.NOTE)
diagnoses_list = pd.read_csv('/word_to_vec_code/D_ICD_DIAGNOSES.csv')
diagnoses_list['LONG_TITLE'] = diagnoses_list['LONG_TITLE'].apply(lambda x: re.sub(r'[,\[\]#\'"]', '', x))
diagnoses_list['LONG_TITLE'] = diagnoses_list['LONG_TITLE'].apply(lambda x: x.replace('-', ' '))
diagnoses_list['LONG_TITLE'] = diagnoses_list['LONG_TITLE'].apply(lambda x: x.lower())
diag = list(diagnoses_list.LONG_TITLE)
sentence_list = []

for sentence in sentences:
    word_list = []
    for word in sentence.strip().split(' '):
        if word in vectors.key_to_index:
            word_list.append(Word(word, vectors[word]))
    sentence_list.append(Sentence(word_list))

# apply single sentence word embedding
sentence_vector_lookup = dict()
sentence_vectors = sentence_to_vec(sentence_list, embedding_size)  # all vectors converted together
if len(sentence_vectors) == len(sentence_list):
    for i in range(len(sentence_vectors)):
        # map: text of the sentence -> vector
        sentence_vector_lookup[i] = sentence_vectors[i]

cosent = defaultdict(list)

for note1, vector1 in sentence_vector_lookup.items():
    for note2, vector2 in sentence_vector_lookup.items():
        sim = 1 - distance.cosine(vector1, vector2)
        cosent[note1].append(sim)

diagnoses = []
drop = []
diag_new = diag.copy()
for i, sentence in enumerate(diag):
    word_list = []
    for word in sentence.strip().split(' '):
        if word in vectors.key_to_index:
            word_list.append(Word(word, vectors[word]))

    if len(word_list) > 0:
        diagnoses.append(Sentence(word_list))
    else:
        drop.append(i)
        diag_new.remove(diag[i])
diagnoses_list.drop(drop, inplace=True)
diag = diag_new.copy()

# apply single sentence word embedding
diag_vector_lookup = dict()
diag_vectors = sentence_to_vec(diagnoses, embedding_size)  # all vectors converted together
if len(diag_vectors) == len(diagnoses):
    for i in range(len(diag_vectors)):
        # map: text of the sentence -> vector
        diag_vector_lookup[diagnoses[i]] = diag_vectors[i]

sims = defaultdict(list)

for note, vector1 in sentence_vector_lookup.items():
    for diagnosis, vector2 in diag_vector_lookup.items():
        sim = l2_dist(vector1, vector2)  # 1 - distance.cosine(vector1, vector2)
        sims[note].append(sim)

top5 = []
for note, vector1 in sims.items():
    top_index = sorted(range(len(sims[note])), key=lambda i: sims[note][i])[:5]
    # top_index = sorted(range(len(sims[note])), key=lambda i: sims[note][i])[-5:]
    top = []
    for d in top_index:
        top.append([diagnoses_list.loc[d].ICD9_CODE, diag[d]])
    top5.append(top)

cosine = defaultdict(list)

for note, vector1 in sentence_vector_lookup.items():
    for diagnosis, vector2 in diag_vector_lookup.items():
        sim = 1 - distance.cosine(vector1, vector2)
        cosine[note].append(sim)

top5_cos = []
for note, vector1 in cosine.items():
    # print(note, max(sims[note])) #, sims[note].index(max(sims[note])))
    # print(note, min(cosine[note]), max(cosine[note]),
    #       cosine[note].index(max(cosine[note])))  # , sims[note].index(max(sims[note])))
    top_index = sorted(range(len(cosine[note])), key=lambda i: cosine[note][i])[-5:]
    top = []
    for d in top_index:
        top.append([diagnoses_list.loc[d].ICD9_CODE, diag[d]])
    top5_cos.append(top)

print(sorted(range(len(sims[200])), key=lambda i: sims[200][i])[:10])
print(sorted(range(len(cosine[0])), key=lambda i: cosine[0][i])[-10:])
top5df = pd.DataFrame(top5, columns=['top1', 'top2', 'top3', 'top4', 'top5'])
top5df = pd.concat([notes[['HADM_ID', 'NOTE']], top5df], axis=1)
top5df.to_csv('top5diagnoses_l2dist.csv')

top5_cosdf = pd.concat([notes[['HADM_ID', 'NOTE']], top5df], axis=1)
top5_cosdf.to_csv('top5diagnoses_cosine.csv')

patient_diagnoses = pd.read_csv('/word_to_vec_code/DIAGNOSES_ICD.csv')
print(patient_diagnoses[patient_diagnoses.HADM_ID == 187637])


print(top5df[top5df.HADM_ID == 187637])