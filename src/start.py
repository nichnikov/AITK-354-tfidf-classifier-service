import os
from src.texts_processing import TextsTokenizer
from src.data_types import Parameters
from src.utils import group_by_lbs
from src.config import (PROJECT_ROOT_DIR, 
                        parameters)
import pandas as pd
from gensim.similarities import MatrixSimilarity
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from itertools import chain


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

texs = list(etalons_df["text"])
tokens = tokenizer(texs)

dct = Dictionary(tokens)
texts_by_groups_tokenized = [[x for x in chain(*tokenizer(txs))] for grp, txs in texts_by_groups]
corpus = [dct.doc2bow(item) for item in texts_by_groups_tokenized]

tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

index = MatrixSimilarity(tfidf[corpus],  num_features=len(dct))

'''
for num, d in enumerate(test_dicts_):
   t = time.time()
   text = d["query"]
   test_tokens = tokenizer([text])
   test_corpus = dct.doc2bow(test_tokens[0])
   test_vector = tfidf[test_corpus]
   sims = index[test_vector]
   tfidf_tuples = [(num, scr) for num, scr in enumerate(list(sims))]
   tfidf_best = sorted(tfidf_tuples, key=lambda x: x[1], reverse=True)[0]
   d["class"] = tfidf_best[0]
   d["tfIdf_score"] = tfidf_best[1]
   print(num, text, time.time() - t)

results_df = pd.DataFrame(test_dicts_)
print(results_df)
results_df.to_csv(os.path.join("data", "test_result_tfidf.csv"), sep="\t", index=False)

'''