from fastT5 import get_onnx_model, get_onnx_runtime_sessions, OnnxT5
from transformers import AutoTokenizer
from pathlib import Path
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str


path_to_model = "./marc"
model_name = Path(path_to_model).stem

path_to_encoder = os.path.join(
    path_to_model, f"{model_name}-encoder-quantized.onnx")
path_to_decoder = os.path.join(
    path_to_model, f"{model_name}-decoder-quantized.onnx")
path_to_init_decoder = os.path.join(
    path_to_model, f"{model_name}-init-decoder-quantized.onnx")

model_paths = path_to_encoder, path_to_decoder, path_to_init_decoder
model_sessions = get_onnx_runtime_sessions(model_paths)

model = OnnxT5(path_to_model, model_sessions)
tokenizer = AutoTokenizer.from_pretrained(path_to_model)


def answer_question(question, m, t):
    encoded_text = t.encode(question, return_tensors="pt").cuda()
    model_output = m.generate(
        encoded_text, do_sample=True, top_p=0.9, max_length=512).cpu()
    answer = t.decode(model_output[0], skip_special_tokens=True)
    return answer


@app.get("/")
def index():
    return {"message": "Welcome to M.A.R.C API"}


@app.post("/solve-math-problem", response_model=QuestionResponse)
def solve_math_problem(request: QuestionRequest):
    question = request.question
    answer = answer_question(question, model, tokenizer)
    return QuestionResponse(answer)
