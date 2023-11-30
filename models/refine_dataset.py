import json
from utils.prompt_generation import refine_passage
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import spacy
nlp = spacy.load("en_core_web_sm")

# sg_parser tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg", local_files_only=True)
from nltk.corpus import wordnet as wn

if __name__ == '__main__':
    LLaVA_QG = json.load(open('results/LLaVA_question_checks.json'))
    evaluation_results = json.load(open('results/bertmatching_evaluation.json'))
    object_evaluation_results = json.load(open('results/object_bertmatching_evaluation.json'))

    hallunation_instances = 0
    total_instances = 0
    hallunation_sentences = 0
    total_sentences = 0
    refined_llava_result = {"annotations": []}
    

    for image_id in tqdm(evaluation_results):
        passage = LLaVA_QG[image_id][0]['description'] # original_descriptions
        hallucination_phrases = []
        gpt_sentences = [i if i[-1] == '.' else i+'.' for i in passage.split('. ')]
        is_hallucination = False
        
        # object hallucination evaluation
        hallucination_objects = object_evaluation_results[image_id]["hallucination_objects"]
        # phrase hallucination evaluation
        for info in evaluation_results[image_id]:
            if (info['score1']+info['score2']+info['score3'])/3<0.5: # consistency threshold
                hallunation_instances += 1
                is_hallucination = True
                hallucination_phrases.append(info['reference'])
            total_instances += 1
        if is_hallucination:
            hallunation_sentences += 1
        total_sentences += 1
        hallucination_phrases.extend(hallucination_objects)
        if len(hallucination_phrases) > 0:
            refined_passage = refine_passage(passage, hallucination_phrases)
        else:
            refined_passage = passage

        refined_llava_result['annotations'].append({
            "image_id": image_id,
            "caption": refined_passage
        })
    print(len(refined_llava_result['annotations']))
    json.dump(refined_llava_result, open('results/refined_LLaVA_cap.json', 'w'))

