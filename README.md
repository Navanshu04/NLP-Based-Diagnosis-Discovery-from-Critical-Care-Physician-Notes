# NLP-Based-Diagnosis-Discovery-from-Critical-Care-Physician-Notes

## Introduction 

Over the past 20 years, the use of Electronic Health Records (EHRs) has rapidly increased and with this growth, developers have improved EHR capabilities. A recent development in the healthcare industry has shifted the responsibility of medical billing coding from dedicated professionals to healthcare providers, leaving little to no training for providers on proper coding. This lack of training results in missed opportunities where additional codes could have been applied but were not identified, leading to lower billing and reimbursement levels. Natural Language Processing (NLP) has emerged as a potential solution to identify additional billing codes by allowing algorithms to analyze documentation and identify diagnoses not included in the provider's coding. In particular, NLP could be useful in critical care settings where documentation is often brief but patients have complex conditions. While numerical or categorical data can be easily analyzed, free-text data is more challenging but can be effectively processed using NLP. Thus, the purpose of this report is to investigate whether an NLP algorithm can be developed to identify potential additional diagnoses for critical care patients, using narrative portions of physician documentation including admission, progress, and discharge notes.

## Dataset

The MIMIC-III Clinical Database, specifically the NOTEEVENTS and DIAGNOSIS_ICD tables, was utilized in this study. The text field in NOTEEVENTS was the primary data source for natural language processing algorithms, which were applied to analyze physician documentation for ICU patients. To reduce data duplication, progress notes within one hour of each other were discarded, leaving around 60,000 records. The assessment and plan portion reported by physicians were focused on by searching for the phrase "Assessment and Plan" within the text column, resulting in 57,592 records. These notes were cleaned up to remove special characters and punctuations, except for the period sign. Diagnoses for a specific patient were extracted from the DIAGNOSIS_ICD table using subject_id and hadm_id as a composite key. The short_title and long_title from the D_ICD_DIAGNOSES table were used to extract the name of the diagnosis for a given icd9_code in the DIAGNOSIS_ICT table. Only diagnoses reported for patients with physician notes were selected, resulting in 4486 unique diagnoses.

## Methods

To address the two problem statements, we randomly sampled 1% of physician notes, which amounted to about 500 notes, to facilitate data processing with limited time and resources. Our processing pipeline involves three steps:
a)	Entity Extraction: Physician notes contain multiple sentences, each with information specific to a diagnosis. Hence, we split the notes into sentences and used MedSpaCy, a clinical NLP toolkit built with spaCy, for clinical sentence segmentation. Although the notes were also segmented into noun phrases, we concatenated all of them into a single sentence to prevent most similarity matching results from appearing random.
b)	Sentence Embeddings Generation: We extracted word vectors for each sentence within a note and each diagnosis description (long_title) using pre-trained word embeddings from BioWordVec and an extension to BioWordVec trained on the MIMIC-III dataset (called BioWordVec-MIMIC). We generated sentence embeddings by combining the word vectors using an open-source implementation of a method proposed by [5]. Additionally, we generated sentence embeddings using the BioSentVec model. We compared the results of using these three embeddings.
c)	Similarity Matching: We compared the sentence embeddings of the notes and diagnoses using cosine similarity and Euclidean distance, two commonly used similarity metrics. We generated the top 5 diagnoses using each similarity metric. For diagnosis validation, we compared the sentence embeddings of the notes only to the confirmed diagnoses for a given patient. However, for diagnosis discovery, we compared all possible diagnoses to identify additional potential diagnoses.

## Conclusion 

In this study, we aimed to use NLP techniques to identify additional diagnoses from physician notes in the MIMIC-III dataset. We worked on a random sample of 1% of the notes, extracting sentences and generating sentence embeddings, which were then compared to a list of diagnoses to find the 5 most similar diagnoses for each sentence. We first tested the pipeline on known diagnoses and found that the simpler BioWordVec model produced results comparable to or better than the more complex BioWordVec model trained on PubMed and MIMIC-III. However, when we moved on to diagnose discovery, we found that the results were not relevant to the given sentence, and instead, the most common diagnoses in the training phase of the model dominated the results. We plan to continue working on this problem using deep learning models and larger data sets to improve our results.
