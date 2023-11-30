import json
from PIL import Image
import torch
import tqdm
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
claim_tokenizer = AutoTokenizer.from_pretrained('khhuang/zerofec-qa2claim-t5-base')
claim_model = AutoModelForSeq2SeqLM.from_pretrained('khhuang/zerofec-qa2claim-t5-base')
def format_inputs(question: str, answer: str):
    return f"{answer} \\n {question}"

def mllm_experts_load(model_id, device):
    print('Initializing MLLM expert for evaluation')
    vis_processors = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device) 
    print('Initialized MLLM experts!!!!')
    return model, vis_processors

def candidate_answers_generation(LLaVA_data_infos, LLaVA_QG, model, processors, device, output_file):
    image_root = '/home/qifan/datasets/coco/train2017/'
    mllm_answers = dict()
    for detail in tqdm.tqdm(LLaVA_data_infos):
        image_qa_pairs = LLaVA_QG[detail['id']]
        image_file = image_root + detail['image']
        image_pil = Image.open(image_file).convert("RGB")

        # BLIP2 evaluation
        mllm_answers[detail['id']] = []
        for image_qa_pair in image_qa_pairs:
            questions = image_qa_pair['llm_questions'][0].split('\n')
            references = image_qa_pair['answers']
            for question, reference in zip(questions, references):
                question = question.split('. ')[-1]
                inputs = processors(image_pil, question, return_tensors="pt").to(device, torch.float16)
                out = model.generate(**inputs)
                blip2_candidate_answer = processors.decode(out[0], skip_special_tokens=True)
                blip2_candidate_answer = blip2_candidate_answer.strip()

                text = format_inputs(question, blip2_candidate_answer)
                input_ids = claim_tokenizer(text, return_tensors="pt").input_ids
                generated_ids = claim_model.generate(input_ids, max_length=32, num_beams=4)
                reference_output = claim_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                current_instructblip_answer = {
                    "question": question,
                    "reference": reference,
                    "pred_answer": reference_output,
                }

                mllm_answers[detail['id']].append(current_instructblip_answer)
    json.dump(mllm_answers, open(output_file, 'w'))


if __name__ == '__main__':
    LLaVA_QG = json.load(open('results/LLaVA_question_checks.json'))
    LLaVA_data_infos = json.load(open('datasets/detail_23k.json'))
    output_file = 'results/answer_checks_blip2.json'
    model_id = "Salesforce/blip2-flan-t5-xl"
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    model, processors = mllm_experts_load(model_id, device)
    candidate_answers_generation(LLaVA_data_infos, LLaVA_QG, model, processors, device, output_file)
