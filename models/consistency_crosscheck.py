import evaluation_utils
import json
import tqdm

blip2answers = json.load(open('results/answer_checks_blip2.json'))
instructblip_answers = json.load(open('results/answer_checks_instructblip.json'))
minigpt4_answers = json.load(open('results/answer_checks_minigpt4.json'))
DIGIT_LIST = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', '2', '3', '4', '5', '6', '7', '8', '9', '10']

if __name__ == '__main__':
    hallucination_num = 0
    chatgpt_responses = []
    bertmatching = dict()

    question_list = []
    reference_list = []
    candidate_answer1_list = []
    candidate_answer2_list = []
    candidate_answer3_list = []
    coresponding_sentence_list = []
    imageid_list = []
    question_type = 'automatic'

    for image_id in tqdm.tqdm(blip2answers):
        blip2_qa_pairs = blip2answers[image_id]
        instructblip_qa_pairs = instructblip_answers[image_id]
        minigpt4_qa_pairs = minigpt4_answers[image_id]
        evaluation_results = []
        for blip2_qa_pair, instructblip_qa_pair, minigpt4_qa_pair in zip(blip2_qa_pairs, instructblip_qa_pairs, minigpt4_qa_pairs):
            question = blip2_qa_pair['question']
            reference = blip2_qa_pair['reference']
            coresponding_sentence = instructblip_qa_pair['current_sentence']
            candidate_answer1 = blip2_qa_pair['pred_answer']
            candidate_answer2 = instructblip_qa_pair['pred_answer']
            candidate_answer3 = minigpt4_qa_pair['pred_answer']
            for d in DIGIT_LIST:
                if d in reference and d not in reference:
                    reference.replace(d, 'serveral')

            question_list.append(question)
            reference_list.append(reference)
            candidate_answer1_list.append(candidate_answer1)
            candidate_answer2_list.append(candidate_answer2)
            candidate_answer3_list.append(candidate_answer3)
            imageid_list.append(image_id)
            coresponding_sentence_list.append(coresponding_sentence)

    total_score1 = evaluation_utils.evaluate_example(
                question_list,
                reference_list,
                candidate_answer1_list,
                question_type=question_type)
    total_score2 = evaluation_utils.evaluate_example(
                question_list,
                reference_list,
                candidate_answer2_list,
                question_type=question_type)
    total_score3 = evaluation_utils.evaluate_example(
                question_list,
                reference_list,
                candidate_answer3_list,
                question_type=question_type)

    instance_hallucination = 0
    sentence_hallucination = dict()
    for question, reference, candidate_answer1, candidate_answer2, candidate_answer3, score1, score2, score3, sentence, image_id in zip(question_list, reference_list, candidate_answer1_list, candidate_answer2_list, candidate_answer3_list, total_score1, total_score2, total_score3, coresponding_sentence_list, imageid_list):
        evaluation_result = {
            "question": question,
            "reference": reference,
            "candidate_answer1": candidate_answer1,
            "candidate_answer2": candidate_answer2,
            "candidate_answer3": candidate_answer3,
            "score1": score1,
            "score2": score2,
            "score3": score3,
            "corresponding_sentence": sentence
                # "LLM_response": response1
        }
        if (score1+score2+score3)/3 < 0.5: # consistency threshold
            instance_hallucination += 1
            sentence_hallucination[image_id] = 1
        if image_id in bertmatching:
            bertmatching[image_id].append(evaluation_result)
        else:
            bertmatching[image_id] = [evaluation_result]
        # object bertmatching evaluation
    json.dump(bertmatching, open('results/bertmatching_evaluation.json', 'w'))
    # hallucination statistics
    print(instance_hallucination, len(total_score1))  
    print(len(sentence_hallucination), len(blip2answers))  


