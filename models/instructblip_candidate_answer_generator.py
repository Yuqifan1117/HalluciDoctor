import argparse

from utils.utils import generate_textgraph
import json
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch
import tqdm
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
DIGIT_LIST = ['three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '.']


# scene graph parser for valid information extraction
sng_tokenizer = AutoTokenizer.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg", local_files_only=True)
sng_model = AutoModelForSeq2SeqLM.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg", local_files_only=True)

def format_inputs(question: str, answer: str):
    return f"{answer} \\n {question}"


def mllm_experts_load(model_id, device):
    print('Initializing MLLM expert for evaluation')
    model, vis_processors, _ = load_model_and_preprocess(name=model_id, model_type="vicuna7b", is_eval=True, device=device)
    return model, vis_processors

def candidate_answers_generation(LLaVA_data_infos, LLaVA_QG, model, processors, device, output_file):
    image_root = '/home/qifan/datasets/coco/train2017/'
    instructblip_mllm_answers = dict()
    for detail in tqdm.tqdm(LLaVA_data_infos):
        image_qa_pairs = LLaVA_QG[detail['id']]
        image_file = image_root + detail['image']
        image_pil = Image.open(image_file).convert("RGB")

        # InstructBLIP evaluation
        instructblip_mllm_answers[detail['id']] = []
        image = processors["eval"](image_pil).unsqueeze(0).to(device)
        for image_qa_pair in image_qa_pairs: 
            context = image_qa_pair['description']
            gpt_sentences = [i if i[-1] == '.' else i+'.' for i in context.split('. ')] # sub-sentence level
            sentence2graph = dict()
            for s in gpt_sentences:
                textgraph = generate_textgraph(s, sng_tokenizer, sng_model, device)
                sentence2graph[s] = []
                for relation in textgraph:
                    sentence2graph[s].append(' '.join(relation))
            if image_qa_pair['llm_questions'] != '':
                questions = image_qa_pair['llm_questions'][0].split('\n')
            else:
                continue
            references = image_qa_pair['answers']
            for question, reference in zip(questions, references):
                question = question.split('. ')[-1]
                for s in sentence2graph:
                    if reference in sentence2graph[s] or reference.split(',')[0] in sentence2graph[s]:
                        current_sentence = s
                prompt = question
                candidate_answer = model.generate({"image": image, "prompt": prompt},
                            length_penalty=float(1),
                            repetition_penalty=float(1.5),
                            num_beams=5,
                            max_length=32,
                            min_length=1,
                            top_p=0.9,
                            use_nucleus_sampling=False)[0]
                candidate_answer = candidate_answer.strip()
                if '\n' in candidate_answer:
                    candidate_answer = candidate_answer.split('\n')[0]
                if candidate_answer.split('. ')[0] != '' and candidate_answer.split('. ')[0][-1] not in DIGIT_LIST:
                    candidate_answer = candidate_answer.split('. ')[0]
                elif candidate_answer.split('. ')[0] != '' and len(candidate_answer.split('. ')) > 1:
                    candidate_answer = candidate_answer.split('. ')[1]
                if candidate_answer != '' and candidate_answer[-1] != '.':
                    processed_candidate_answer = candidate_answer + '.'
                else:
                    processed_candidate_answer = candidate_answer



                current_instructblip_answer = {
                    "question": question,
                    "reference": reference,
                    "current_sentence": current_sentence if current_sentence != None else '',
                    "pred_answer": processed_candidate_answer,
                }


                instructblip_mllm_answers[detail['id']].append(current_instructblip_answer)
    json.dump(instructblip_mllm_answers, open(output_file, 'w'))
                    


if __name__ == '__main__':
    LLaVA_QG = json.load(open('results/LLaVA_question_checks.json'))
    LLaVA_data_infos = json.load(open('datasets/detail_23k.json'))
    output_file = 'results/answer_checks_instructblip.json'
    model_id = "blip2_vicuna_instruct"
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    model, processors = mllm_experts_load(model_id, device)
    candidate_answers_generation(LLaVA_data_infos, LLaVA_QG, model, processors, device, output_file)
