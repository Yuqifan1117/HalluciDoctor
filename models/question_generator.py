import random
import torch
from tqdm import tqdm
import json
import spacy
from utils.prompt_generation import qgqa_generation
from utils.utils import generate_textgraph
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# scene graph parser for valid information extraction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
sng_tokenizer = AutoTokenizer.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg", local_files_only=True)
sng_model = AutoModelForSeq2SeqLM.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg", local_files_only=True)
# datasets
details = json.load(open('datasets/detail_23k.json'))
coco_category = json.load(open('utils/coco_category.json'))
valid_nouns = []
nlp = spacy.load("en_core_web_sm")
DIGIT_LIST = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'yes', 'no']

for k in coco_category:
    valid_nouns.append(coco_category[k])
# visual genome valid objects
vg_category = ["tree", "window", "shirt", "building", "person", "table", "car", "door", "light", "fence", "chair", "people", "plate","glass", "jacket", "sidewalk", "snow", "flower", "hat", "bag", "track", "roof", "umbrella", "helmet", "plant", "train", "bench", "box", "food", "pillow", "bus", "bowl", "horse", "trunk", "clock", "mountain", "elephant", "giraffe", "banana", "house", "cabinet", "hill", "dog", "book", "bike", "coat", "glove", "zebra", "bird", "motorcycle", "lamp", "cow", "skateboard", "surfboard", "beach", "sheep", "kite", "cat", "pizza", "bed", "bear", "windshield", "towel", "desk"]
for tag in vg_category:
    if tag not in valid_nouns:
        valid_nouns.append(tag)


def generate_tags(raw_text):
    # generate specific categories in the caption by spacy
    nlp = spacy.load("en_core_web_sm")
    tags = {'nouns':[], 'adj':[], 'verb':[]}
    words_list = nlp(raw_text)
    if len(words_list) < 2:
        return None
    for i in range(1, len(words_list)-2):
        last_token = words_list[i-1]
        token = words_list[i]
        next_token = words_list[i+1]
        next_next_token = words_list[i+2]
        if token.pos_ == 'ADJ' and next_token.pos_ == 'NOUN':
            tags['adj'].append(token.text.lower()+' '+next_token.text.lower())
        elif token.pos_ == 'NOUN' and next_token.pos_ != 'ADP':
            tags['nouns'].append(token.text.lower())
        elif token.pos_ == 'VERB'and (next_token.pos_ == 'NOUN' or (next_token.pos_ == 'DET' and next_next_token.pos_ == 'NOUN')): # how to distinguish whether it can be shown in the image
            if next_token.pos_ == 'DET':
                tags['verb'].append(token.text.lower() + ' ' + next_token.text.lower()+ ' ' + next_next_token.text.lower())
            else:
                tags['verb'].append(token.text.lower() + ' ' + next_token.text.lower())
    if words_list[-1].pos_ == 'NOUN':
        tags['nouns'].append(words_list[-1].text.lower())
    if words_list[-2].pos_ == 'NOUN':
        tags['nouns'].append(words_list[-2].text.lower())
    if words_list[-2].pos_ == 'ADJ' and words_list[-1].pos_ == 'NOUN':
        tags['adj'].append(words_list[-2].text.lower() + ' ' + words_list[-1].text.lower())
    return tags

def question_generation(question_type, context, instruct, sub_sentence=True):
    QA_response = [] # return to save QA pairs
    if question_type == 'rule':
        if sub_sentence:
            gpt_sentences = [i if i[-1] == '.' else i+'.' for i in context.split('. ')] # sub-sentence level
            gpt_sentences.append(instruct)
        else:
            gpt_sentences = [instruct + ' ' +context]
        nouns = []
        verb = []
        activity = []
        attribute = []
        fusion_attribute = dict()
        fusion_activity = dict()
        for gpt_sentence in gpt_sentences:
            textgraph = generate_textgraph(gpt_sentence, sng_tokenizer, sng_model, device)
            attributes_verb = ['is', 'are', 'have', 'has'] # attribute predicates 
            for relation in textgraph:
                if len(relation) != 3:
                    continue
                if relation[1] not in attributes_verb:
                    if relation[0] not in nouns:
                        nouns.append(relation[0])
                    if relation[1] not in verb:
                        verb.append(relation[1])
                    if relation[0] in valid_nouns or nlp(relation[0])[0].lemma_ in valid_nouns:
                        activity.append(relation)
                        if ' '.join([nlp(relation[0])[0].lemma_, nlp(relation[2])[0].lemma_]) in fusion_activity and relation not in fusion_activity[' '.join([nlp(relation[0])[0].lemma_, nlp(relation[2])[0].lemma_])]:
                            fusion_activity[' '.join([nlp(relation[0])[0].lemma_, nlp(relation[2])[0].lemma_])].append(relation)
                        else:
                            fusion_activity[' '.join([nlp(relation[0])[0].lemma_, nlp(relation[2])[0].lemma_])] = [relation]
                else:
                    if relation[0] not in nouns:
                        nouns.append(relation[0])
                    if relation[2] not in DIGIT_LIST and nlp(relation[2])[0].pos_ != "NOUN":
                        attribute.append(relation)
                        if ' '.join([relation[0], relation[1]]) in fusion_attribute and relation not in fusion_attribute[' '.join([relation[0], relation[1]])]:
                            fusion_attribute[' '.join([relation[0], relation[1]])].append(relation)
                        else:
                            fusion_attribute[' '.join([relation[0], relation[1]])] = [relation]

        # question_incontext_prompts = "\nPlease give me a meaningful and answerable question corresponding to the following answer based on the given context to help me understand the context.\n"
        valid_qa_pairs = {
            "description": context,
            "questions": [],
            "answers": []
        }
        valid_answers = []
        for k in fusion_attribute:
            valid_answers.append(k + ' ' + ', '.join([r[2] for r in fusion_attribute[k]]))
        for k in fusion_activity:
            i = random.randint(0, len(fusion_activity[k])-1)
            valid_answers.append(' '.join(fusion_activity[k][i]))
        valid_qa_pairs["llm_questions"] = qgqa_generation(context, valid_answers)
        pred_questions = valid_qa_pairs["llm_questions"][0].split('\n')
        for pred_question, answer in zip(pred_questions, valid_answers):
            pred_question = pred_question.split('. ')[-1]
            valid_qa_pairs['questions'].append(pred_question)
            valid_qa_pairs['answers'].append(answer)
        QA_response.append(valid_qa_pairs)
    
    return QA_response

if __name__ == '__main__':
    # question generation
    total_QA_response = dict()
    for detail in tqdm(details):
        # original question
        instruction = detail['conversations'][0]['value'].replace('<image>', '').replace('\n', '')
        gpt_result = detail['conversations'][1]['value'].replace('\n\n',' ') 
        image_id = detail['id']
        QA_response = question_generation('rule', gpt_result, instruction, sub_sentence=True)
        total_QA_response[image_id] = QA_response
    json.dump(total_QA_response, open('results/LLaVA_question_checks.json','w'))