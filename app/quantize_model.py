from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config
from fastT5 import export_and_get_onnx_model, generate_onnx_representation, quantize
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

path_to_model = f"{__location__}/model_and_tokenizer/model"
path_to_tokenizer = f"{__location__}/model_and_tokenizer/tokenizer"


tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer)
model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model)


def answer_question(question):
    encoded_text = tokenizer.encode(question, return_tensors="pt")
    model_output = model.generate(
        encoded_text, do_sample=True, top_p=0.9, max_length=512
    )
    answer = tokenizer.decode(model_output[0], skip_special_tokens=True)
    return answer


def quantize_model():
    model_path = f"{__location__}/model_and_tokenizer"
    onnx_model_paths = generate_onnx_representation(model_path)

    quantize_model_paths = quantize(onnx_model_paths)

    tokenizer_onnx = AutoTokenizer.from_pretrained(model_path)
    config = T5Config.from_pretrained(model_path)

    tokenizer_onnx.save_pretrained(f"{__location__}/models/")
    config.save_pretrained(f"{__location__}/models/")


a = answer_question(
    "John has 2 apples and gives away 1. How many apples does John have now?"
)
print(a)
