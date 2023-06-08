""""""
import os
import uvicorn
from fastapi import FastAPI
from src.start import (classifier, 
                       pubs)
from src.config import (logger,
                        parameters)
from src.data_types import SearchData

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI(title="ExpertBotFastText")


@app.post("/api/search")
async def search(data: SearchData):
    """searching etalon by  incoming text"""
    logger.info("searched pubid: {} searched text: {}".format(str(data.pubid), str(data.text)))
    if data.pubid in pubs:
        try:
            logger.info("searched text without spellcheck: {}".format(str(data.text)))
            result = await classifier.searching(str(data.text))
            return result
        except Exception:
            logger.exception("Searching problem with text {} in pubid {}".format(str(data.text), str(data.pubid)))
            return {"templateId": 0, "templateText": "", "text": str(data.text)}
    else:
        return {"templateId": 0, "templateText": "", "text": str(data.text)}

if __name__ == "__main__":
    # uvicorn.run(app, host=service_host, port=service_port)
    uvicorn.run(app, host="0.0.0.0", port=8080)
