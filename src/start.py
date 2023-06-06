import os
import pandas as pd
from itertools import chain
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.similarities import MatrixSimilarity
from src.classifiers import TechSupportClassifier
from src.texts_processing import TextsTokenizer
from src.utils import group_by_lbs
from src.config import (PROJECT_ROOT_DIR, 
                        parameters)


stopwords = []
if parameters.stopwords_files:
    for filename in parameters.stopwords_files:
        root = os.path.join(PROJECT_ROOT_DIR, "data", filename)
        stopwords_df = pd.read_csv(root, sep="\t")
        stopwords += list(stopwords_df["stopwords"])

tokenizer = TextsTokenizer()
tokenizer.add_stopwords(stopwords)

etalons_df = pd.read_csv(os.path.join("data", "etalons.csv"), sep="\t")
groups_texts = list(zip(etalons_df["label"], etalons_df["query"]))
texts_by_groups = sorted(list(group_by_lbs(groups_texts)), key=lambda x: x[0])

answers_by_labels = {l: a for l, a in set((lb, ans) for lb, ans in 
                                          zip(etalons_df["label"], etalons_df["templateText"]))}

texs = list(etalons_df["query"])
tokens = tokenizer(texs)

dct = Dictionary(tokens)
texts_by_groups_tokenized = [[x for x in chain(*tokenizer(txs))] for grp, txs in texts_by_groups]
corpus = [dct.doc2bow(item) for item in texts_by_groups_tokenized]

tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

index = MatrixSimilarity(tfidf[corpus],  num_features=len(dct))

classifier = TechSupportClassifier(
                tokenizer=tokenizer, 
                parameters=parameters, 
                gensim_dict=dct, 
                tfidf_model=tfidf, 
                gensim_index=index,
                answers=answers_by_labels)

pubs_df = pd.read_csv(os.path.join("data", "pubs.csv"), sep="\t")
pubs = list(pubs_df["pubid"])