from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
import random
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URI"), serverSelectionTimeoutMS=5000)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str
    use_cache: bool = True


class QuestionResponse(BaseModel):
    answer: str
    possible_solutions: List[str]


class TextToQuestionRequest(BaseModel):
    context: str
    answer: str
    number_of_questions: int = 1


class TextToQuestionResponse(BaseModel):
    question: List[str]


path_to_math_model = f"{__location__}/math/model"
path_to_math_tokenizer = f"{__location__}/math/tokenizer"

path_to_question_gen_model = f"{__location__}/question_generation/model"
path_to_question_gen_tokenizer = f"{__location__}/question_generation/tokenizer"

path_to_verifier_model = f"{__location__}/solution_verifier/model"
path_to_verifier_tokenizer = f"{__location__}/solution_verifier/tokenizer"

math_tokenizer = AutoTokenizer.from_pretrained(path_to_math_tokenizer)
math_model = AutoModelForSeq2SeqLM.from_pretrained(path_to_math_model)

question_gen_tokenizer = AutoTokenizer.from_pretrained(path_to_question_gen_tokenizer)
question_gen_model = AutoModelForSeq2SeqLM.from_pretrained(path_to_question_gen_model)

verifier_tokenizer = AutoTokenizer.from_pretrained(path_to_verifier_tokenizer)
verifier_model = AutoModelForSeq2SeqLM.from_pretrained(path_to_verifier_model)


class MathSolver:
    @staticmethod
    def clean_up_calculations(text):
        CALCULATOR_PATTERN = r"(\d+(\s*[-+*/]\s*\d+)*\s*=\s*\d+)\s*>>\s*(\d+)"
        FINAL_ANSWER_NUMBER_PATTERN = r"\d+\S*$"
        NO_SPACE_EQUATION_PATTERN = r"(\d+)([+\-*/])(\d+)"

        # Find All Calculator Notions Of Form "X+Y=Z>>Z"
        calculator_notation = re.findall(CALCULATOR_PATTERN, text)

        if not calculator_notation:
            return text

        correct_answer = None

        for calc_notation in calculator_notation:
            # Replace "X+Y=Z>>Z" To Evaluated Value "X+Y"
            expression, _, final_answer = calc_notation
            notation = expression + ">>" + final_answer
            equation = expression.strip()
            lhs = equation.split("=")
            if not lhs:
                continue
            try:
                correct_answer = eval(lhs[0])
            except:
                correct_answer = lhs[1] if len(lhs) == 2 else 0
            text = text.replace(notation, str(correct_answer))

        # Verify Final Answer Is Same As Final Calculation
        if correct_answer and re.search(FINAL_ANSWER_NUMBER_PATTERN, text):
            text = re.sub(FINAL_ANSWER_NUMBER_PATTERN, str(correct_answer), text)

        # Add Space Between Equations: "X+Y=Z" => "X + Y = Z"
        text = re.sub(NO_SPACE_EQUATION_PATTERN, r"\1 \2 \3", text)

        return text

    @staticmethod
    def extract_correct_boolean(text):
        IS_CORRECT_PATTERN = r"is_correct: (true|false)"

        match = re.search(IS_CORRECT_PATTERN, text, re.IGNORECASE)

        if match:
            return match.group(1).lower() == "true"
        else:
            return False

    @staticmethod
    def answer_question(question):
        question = f"solve: {question}"
        encoded_text = math_tokenizer.encode(question, return_tensors="pt")
        model_output = math_model.generate(
            encoded_text, do_sample=True, top_p=0.9, max_length=256
        )
        answer = math_tokenizer.decode(model_output[0], skip_special_tokens=True)
        answer = MathSolver.clean_up_calculations(answer)
        return answer

    @staticmethod
    def get_majority_answer(possible_solutions):
        try:
            ANSWER_PATTERN = r"The final answer is (\d+)"

            final_answers = [
                re.findall(ANSWER_PATTERN, solution)[0]
                for solution in possible_solutions
                if re.search(ANSWER_PATTERN, solution)
            ]

            answer_count = {
                answer: final_answers.count(answer) for answer in final_answers
            }

            majority_answer = max(answer_count, key=answer_count.get, default=0)

            correct_solutions = [
                solution
                for solution in possible_solutions
                if "The final answer is " + majority_answer in solution
            ]

            return random.choice(correct_solutions)
        except:
            print("MAJORITY_ANSWER COULD NOT BE COMPUTED")
            return random.choice(possible_solutions)

    @staticmethod
    def generate_possible_solutions(question):
        text = f"solve: {question}"

        encoding = math_tokenizer.encode_plus(
            text, max_length=512, padding=True, return_tensors="pt"
        )
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        math_model.eval()

        beam_search_output = math_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=72,
            early_stopping=True,
            num_beams=50,
            num_return_sequences=50,
        )

        return [
            MathSolver.clean_up_calculations(
                math_tokenizer.decode(
                    beam_output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )
            for beam_output in beam_search_output
        ]

    @staticmethod
    def verify_solutions_beam_search(question, answers):
        possible_solutions = []
        for possible_answer in answers:
            formatted_text = f"verify: {possible_answer} question: {question}"
            encoding = verifier_tokenizer.encode_plus(
                formatted_text, max_length=512, padding=True, return_tensors="pt"
            )
            input_ids, attention_mask = (
                encoding["input_ids"],
                encoding["attention_mask"],
            )

            verifier_model.eval()

            beam_search_output = verifier_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=72,
                early_stopping=True,
                num_beams=50,
                num_return_sequences=1,
            )

            verify_output = math_tokenizer.decode(
                beam_search_output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            verify = MathSolver.extract_correct_boolean(verify_output)

            if verify:
                possible_solutions.append(possible_answer)

        return MathSolver.get_majority_answer(possible_solutions), possible_solutions

    @staticmethod
    def verify_solution(question, answers):
        possible_solutions = []
        for possible_answer in answers:
            formatted_text = f"verify: {possible_answer} question: {question}"
            encoded_text = verifier_tokenizer.encode(
                formatted_text, return_tensors="pt"
            )
            model_output = verifier_model.generate(
                encoded_text, do_sample=True, top_p=0.9, max_length=512
            )
            verify_output = verifier_tokenizer.decode(
                model_output[0], skip_special_tokens=True
            )
            verify = MathSolver.extract_correct_boolean(verify_output)

            if verify:
                possible_solutions.append(possible_answer)

        return MathSolver.get_majority_answer(possible_solutions), possible_solutions


class TextToQuestionGenerator:
    def text_to_question(context, answer, number_of_questions):
        text = f"context: {context} answer: {answer} </s>"

        encoding = question_gen_tokenizer.encode_plus(
            text, max_length=512, padding=True, return_tensors="pt"
        )
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        question_gen_model.eval()

        beam_search_output = question_gen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=72,
            early_stopping=True,
            num_beams=5,
            num_return_sequences=min(number_of_questions, 4),
        )

        return [
            question_gen_tokenizer.decode(
                beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for beam_output in beam_search_output
        ]


@app.get("/")
def index():
    return {"message": "Welcome to M.A.R.C API"}


@app.post("/solve-math-problem", response_model=QuestionResponse)
def solve_math_problem(request: QuestionRequest):
    try:
        question = request.question.strip().lower()
        use_cache = request.use_cache
        db = client["Solutions"]
        cache = db["cache"]

        # Check If Question Is Cached
        cache.create_index([("question", "text")])
        cached_question = cache.find_one({"$text": {"$search": question}})
        cache_time = datetime.datetime.now() - datetime.timedelta(days=1)

        if (
            use_cache
            and cached_question
            and re.search(cached_question["question"], question, re.IGNORECASE)
            and (cached_question["date"] >= cache_time)
        ):
            return QuestionResponse(
                answer=cached_question["answer"],
                possible_solutions=cached_question["possible_solutions"],
            )

        possible_answers = [MathSolver.answer_question(question) for _ in range(30)]
        answer, possible_solutions = MathSolver.verify_solution(
            question, possible_answers
        )

        cache_result = {
            "question": question,
            "answer": answer,
            "possible_solutions": possible_solutions,
            "date": datetime.datetime.today(),
        }

        if cached_question and (cached_question["date"] < cache_time):
            cache.update_one(
                {"question": question}, {"$set": cache_result}, upsert=False
            )
        else:
            cache.insert_one(cache_result)

        return QuestionResponse(answer=answer, possible_solutions=possible_solutions)
    except Exception as e:
        return QuestionResponse(answer="SERVER ERROR " + str(e), possible_solutions=[])


@app.post("/generate-question", response_model=TextToQuestionResponse)
def generate_question(request: TextToQuestionRequest):
    try:
        context = request.context
        answer = request.answer
        number_of_questions = request.number_of_questions
        question = TextToQuestionGenerator.text_to_question(
            context, answer, number_of_questions
        )
        return TextToQuestionResponse(question=question)
    except Exception as e:
        return TextToQuestionResponse(question="SERVER ERROR " + str(e))
