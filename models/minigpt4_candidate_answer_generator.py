import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================
ckpt = "/home1/yqf/MiniGPT-4/checkpoints/pretrained_minigpt4_7b.pth"

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.ckpt = ckpt
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = [StoppingCriteriaSub(stops=stop_words_ids)]

chatmodel = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')

import json
import random
from tqdm import tqdm
from PIL import Image
import spacy
nlp = spacy.load("en_core_web_sm")
import warnings
warnings.filterwarnings("ignore")
coco_category = json.load(open('utils/coco_category.json'))
LLaVA_QG = json.load(open('results/LLaVA_question_checks.json'))
detail_info = json.load(open('datasets/detail_23k.json'))

# PYTHONPATH=./ python models/minigpt4_candidate_answer_generator.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 3
if __name__ == '__main__':
    chat_state = CONV_VISION.copy()
    minigpt4_mllm_answers = dict()
    for detail in tqdm(detail_info):
        image_qa_pairs = LLaVA_QG[detail['id']]
        image_path = '/home/qifan/datasets/coco/train2017/' + detail['image']
        instruct_image = Image.open(image_path).convert("RGB")
        minigpt4_mllm_answers[detail['id']] = []
        img_list = []
        chat_state.messages = []
        chatmodel.upload_img(image_path, chat_state, img_list)
        image_chat_state = chat_state.copy()
        for image_qa_pair in image_qa_pairs:
            context = image_qa_pair['description']
            questions = image_qa_pair['llm_questions'][0].split('\n')
            references = image_qa_pair['answers']
            for question, reference in zip(questions, references):
                question = question.split('. ')[-1]
                prompt = question
                chatmodel.ask(prompt, chat_state)
                response = chatmodel.answer(conv=chat_state,
                                        img_list=img_list,
                                        num_beams=5,
                                        temperature=1.0,
                                        repetition_penalty=1.0,
                                        max_new_tokens=64,
                                        max_length=300)[0]
                chat_state = image_chat_state.copy()
                response = response.split('</s>')[0] # filter </s>
                response = response.split('</Img>')[-1] # filter the first </Img>
                response = response.split('\n')[0].split('. ')[0].strip()
                current_instructblip_answer = {
                    "question": question,
                    "reference": reference,
                    "pred_answer": response,
                }
                minigpt4_mllm_answers[detail['id']].append(current_instructblip_answer)
    json.dump(minigpt4_mllm_answers, open('results/answer_checks_minigpt4.json', 'w'))

