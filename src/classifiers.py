"""
классификатор KNeighborsClassifier в /home/an/Data/Yandex.Disk/dev/03-jira-tasks/aitk115-support-questions
"""
from src.start import answers_by_labels
from src.data_types import Parameters
from src.texts_processing import TextsTokenizer
from src.config import logger
from collections import namedtuple

# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

class FastAnswerClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self, tokenizer: TextsTokenizer, 
                 parameters: Parameters, 
                 gensim_dict, 
                 tfidf_model, 
                 gensim_index,
                 answers_by_labels: dict):
        self.tkz = tokenizer
        self.prm = parameters
        self.tfidf = tfidf_model
        self.dct = gensim_dict
        self.index = gensim_index
        self.answers = answers_by_labels

    async def searching(self, text: str, pubid: int, score: float):
        """searching etalon by  incoming text"""
        try:
            tokens = self.tkz([text])
            if tokens[0]:
                in_corpus = self.dct.doc2bow(tokens[0])
                in_vector = self.tfidf[in_corpus]
                sims = self.index[in_vector]
                tfidf_tuples = [(num, scr) for num, scr in enumerate(list(sims)) if scr >= score]
                if tfidf_tuples:
                    tfidf_best = sorted(tfidf_tuples, key=lambda x: x[1], reverse=True)[0]
                    best_id = tfidf_best[0]
                    templateText = self.answers[best_id]
                    search_result = {"templateId": best_id, "templateText": templateText}
                    logger.info("search completed successfully with result: {}".format(str(search_result)))
                    return search_result
                else:
                    return {"templateId": 0, "templateText": ""}
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            return {"templateId": 0, "templateText": ""}