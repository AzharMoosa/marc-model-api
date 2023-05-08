from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import os
from fastapi import FastAPI
from pydantic import BaseModel
import re

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str


path_to_model = f"{__location__}/model"
path_to_tokenizer = f"{__location__}/tokenizer"

tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer)
model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model)


def clean_up_calculations(text):
    CALCULATOR_PATTERN = r"\d+[+/*x-]\d+=\d+>>\d+"
    FINAL_ANSWER_NUMBER_PATTERN = r"\d+\S*$"
    NO_SPACE_EQUATION_PATTERN = r"(\d+)([+\-*/])(\d+)"

    # Find All Calculator Notions Of Form "X+Y=Z>>Z"
    calculator_notation = re.findall(CALCULATOR_PATTERN, text)

    if not calculator_notation:
        return text

    correct_answer = None

    for notation in calculator_notation:
        # Replace "X+Y=Z>>Z" To Evaluated Value "X+Y"
        expression = notation.split(">>")
        equation = expression[0].strip()
        lhs = equation.split("=")
        if not lhs:
            continue
        correct_answer = eval(lhs[0])
        text = text.replace(notation, str(correct_answer))

    # Verify Final Answer Is Same As Final Calculation
    if correct_answer and re.search(FINAL_ANSWER_NUMBER_PATTERN, text):
        text = re.sub(FINAL_ANSWER_NUMBER_PATTERN, str(correct_answer), text)

    # Add Space Between Equations: "X+Y=Z" => "X + Y = Z"
    text = re.sub(NO_SPACE_EQUATION_PATTERN, r"\1 \2 \3", text)

    return text


def answer_question(question):
    encoded_text = tokenizer.encode(question, return_tensors="pt")
    model_output = model.generate(
        encoded_text, do_sample=True, top_p=0.9, max_length=256
    )
    answer = tokenizer.decode(model_output[0], skip_special_tokens=True)
    answer = clean_up_calculations(answer)
    return answer


@app.get("/")
def index():
    return {"message": "Welcome to M.A.R.C API"}


@app.post("/solve-math-problem", response_model=QuestionResponse)
def solve_math_problem(request: QuestionRequest):
    try:
        question = request.question
        answer = answer_question(question)
        return QuestionResponse(answer=answer)
    except:
        return QuestionResponse(answer="SERVER ERROR")
