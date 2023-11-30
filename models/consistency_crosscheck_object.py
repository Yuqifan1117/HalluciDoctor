import json
from typing import Dict
from tqdm import tqdm
import torch
from utils.utils import  generate_textgraph
from utils.grounded_sam import load_image, load_model, get_grounding_output
import warnings
warnings.filterwarnings("ignore")
import spacy
nlp = spacy.load("en_core_web_sm")
BOX_TRESHOLD = 0.35     # used in detector api.
TEXT_TRESHOLD = 0.25    # used in detector api.


def filter_objects(obj, target_set):
    if obj in target_set or nlp(obj)[0].lemma_ in target_set:
        return True
    else:
        return False        

def generate_tags(raw_text):
    # generate specific categories in the caption by spacy
    tags = {'nouns':[], 'adj':[], 'verb':[]}
    words_list = nlp(raw_text)
    for i in range(0, len(words_list)-1):
        token = words_list[i]
        next_token = words_list[i+1]
        if token.pos_ == 'NOUN' and next_token.pos_ != 'ADP' and next_token.pos_ != 'NOUN':
            tags['nouns'].append(token.text.lower())
        if token.pos_ == 'NOUN' and next_token.pos_ == 'NOUN':
            tags['nouns'].append(token.text.lower() + ' '+ next_token.text.lower())
    for i in range(0, len(words_list)-1):
        token = words_list[i]
        next_token = words_list[i+1]
        if token.pos_ == 'ADJ' and next_token.pos_ == 'NOUN':
            tags['adj'].append(token.text.lower()+' '+next_token.text.lower())
    for i in range(0, len(words_list)-2):
        token = words_list[i]
        next_token = words_list[i+1]
        next_next_token = words_list[i+2]
        if token.pos_ == 'VERB':
            if next_token.pos_ == 'DET' and next_next_token.pos_ == 'NOUN':
                tags['verb'].append(token.text.lower() + ' ' + next_token.text.lower()+ ' ' + next_next_token.text.lower())
            elif next_token.pos_ == 'NOUN':
                tags['verb'].append(token.text.lower() + ' ' + next_token.text.lower())
    if words_list[-1].pos_ == 'NOUN' and words_list[-2].pos_ != 'NOUN':
        tags['nouns'].append(words_list[-1].text.lower())
    if words_list[-2].pos_ == 'ADJ' and words_list[-1].pos_ == 'NOUN':
        tags['adj'].append(words_list[-2].text.lower() + ' ' + words_list[-1].text.lower())
    if words_list[-2].pos_ == 'VERB' and words_list[-1].pos_ == 'NOUN':
        tags['verb'].append(words_list[-2].text.lower() + ' ' + words_list[-1].text.lower())
    return tags

# sg_parser tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg", local_files_only=True)
from nltk.corpus import wordnet as wn
def synset_list(word):
    a = wn.synsets(word)
    synsets_list = []
    for w in a:
        if w.lemmas()[0].name() not in synsets_list:
            synsets_list.append(w.lemmas()[0].name())
    return synsets_list
if __name__ == '__main__':
    detector_config = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py'
    detector_model = 'checkpoints/groundingdino_swinb_cogcoor.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coco_category = json.load(open('utils/coco_category.json'))
    valid_nouns = []
    for k in coco_category:
        valid_nouns.append(coco_category[k])
    # visual genome valid objects
    vg_category = ["tree", "window", "shirt", "building", "person", "table", "car", "door", "light", "fence", "chair", "people", "plate","glass", "jacket", "sidewalk", "snow", "flower", "hat", "bag", "track", "roof", "umbrella", "helmet", "plant", "train", "bench", "box", "food", "pillow", "bus", "bowl", "horse", "trunk", "clock", "mountain", "elephant", "giraffe", "banana", "house", "cabinet", "hill", "dog", "book", "bike", "coat", "glove", "zebra", "bird", "motorcycle", "lamp", "cow", "skateboard", "surfboard", "beach", "sheep", "kite", "cat", "pizza", "bed", "bear", "windshield", "towel", "desk"]
    for tag in vg_category:
        if tag not in valid_nouns:
            valid_nouns.append(tag)
    dino_model = load_model(detector_config, detector_model, device=device)

    LLaVA_QG = json.load(open('results/LLaVA_question_checks.json'))
    evaluation_results = open('results/bertmatching_evaluation.json')

    refined_llava_result = {"annotations": []}
    hallucination_object_infos = dict()
    for image_id in tqdm(evaluation_results):
        passage = LLaVA_QG[image_id][0]['description']
        gpt_sentences = [i if i[-1] == '.' else i+'.' for i in passage.split('. ')]
        pseudo_extracted_entities = []
        source_entities = []
        for sentence in gpt_sentences:
            textgraph = generate_textgraph(sentence, tokenizer, model, device)
            source_entities.extend(generate_tags(sentence)['nouns'])
            # similar_word
            for relation in textgraph:
                if len(relation) != 3:
                    continue 
                if filter_objects(relation[0], valid_nouns) and relation[0] not in pseudo_extracted_entities:
                    pseudo_extracted_entities.append(relation[0])    
                if filter_objects(relation[2], valid_nouns) and relation[2] not in pseudo_extracted_entities:
                    pseudo_extracted_entities.append(relation[2])
        target_docs = [nlp(text)[0].lemma_ for text in pseudo_extracted_entities]
        for source_str in source_entities:
            source_doc = nlp(source_str)[0].text.lower()
            synsets = synset_list(source_doc)
            for target_str in target_docs:
                if target_str in synsets and source_str not in pseudo_extracted_entities:
                    pseudo_extracted_entities.append(source_str)
        img_path = f'/home/qifan/datasets/coco/train2017/{image_id}.jpg'
        image_source, image = load_image(img_path)
        entity_str = '.'.join(pseudo_extracted_entities)
        boxes, phrases, logits = get_grounding_output(
            model=dino_model,
            image=image,
            caption=entity_str,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            with_logits=False,
            device='cuda'
        )
        detected_phrases = []
        for phrase in phrases:
            for sub_phrase in phrase.split(' '):
                if sub_phrase in pseudo_extracted_entities and sub_phrase not in detected_phrases:
                    detected_phrases.append(sub_phrase)
        hallucination_objects = []
        for n in pseudo_extracted_entities:
            if n not in detected_phrases:
                hallucination_objects.append(n)
        hallucination_object_infos[image_id] = {
            "hallucination_objects": hallucination_objects,
            "detected_objects": detected_phrases,
            "pseudo_extracted_entities": pseudo_extracted_entities
        }
    json.dump(hallucination_object_infos, open('results/object_bertmatching_evaluation.json', 'w'))
