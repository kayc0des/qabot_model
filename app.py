from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from predict import load_model, predict_and_respond, get_response

app = FastAPI()

# Load the model when the app starts
load_model()

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    response: str
    intent: str

@app.post("/qa", response_model=Answer)
async def qa_endpoint(question: Question):
    try:
        intent, response = predict_and_respond(question.text)
        return Answer(response=response, intent=intent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Q&A API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)