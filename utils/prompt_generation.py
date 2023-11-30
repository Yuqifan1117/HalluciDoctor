import requests
from retrying import retry
import openai

@retry(stop_max_attempt_number=10, wait_fixed=2000)
def send_request(data):
    # openai api for completion or chat
    # if use_completion:
    #     data = convert_chat_to_completion(data)
    # try:
    #     response = requests.post(url, headers=HEADER, data=json.dumps(data), verify=False, proxies=PROXY)
    #     response_json = json.loads(response.text)
    #     if "choices" in response_json and len(response_json["choices"]) > 0:
    #         if use_completion:
    #             return response.json()["choices"][0]["text"]
    #         else:
    #             return response.json()["choices"][0]["message"]["content"]
    #     else:
    #         return response_json
    # except requests.exceptions.RequestException as e:
    #     print(f"Error: {e}")

    # chinese api for completion or chat
    openai.api_key = ""
    openai.api_base = ""
    

    try:
        # stream
        messages = data["messages"]
        response = openai.ChatCompletion.create(
            model=data['model'],
            messages=data["messages"],
            temperature=data["temperature"],
            stream=True,
        )        
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                # print(f'收到的完成数据: {completion}')
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                # print(f'流响应数据: {delta_k} = {delta_v}')
                completion[delta_k] += delta_v
        messages.append(completion)  # 直接在传入参数 messages 中追加消息
        content = completion['content']
        return content
    except requests.exceptions.Timeout as e:                                           
        print(f"Timeout Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    return None


# Include questions must be answerable and about the context of the description, including the object types, object actions, object locations, relationships between objects, etc.
def qgqa_generation(caption, answer):
    messages = [{"role": "user", "content": f"{caption}\nPlease give me meaningful and answerable questions corresponding to the following answers based on the given context to help me understand the context. Please ensure that each question doesn't involve 'How many' and is concise to exactly match the corresponding answer.\nAnswer: {answer}"}]
    generation_data = {
        "model": 'gpt-3.5-turbo',
        "messages": messages,
        "temperature": 1.0
    }
    results = send_request(generation_data)
    retries = 0
    while (results == None or 'error' in results) and retries < 3:
        results = send_request(generation_data)
        retries += 1
    if results is not None and 'error' not in results:
        results = results.split('\n\n')
    else:
        return ""
    return results


def refine_passage(passage, hallucination_phrases):
    PROMPT_TEMPLATE='''Given a passage and wrong phrases, you are required to remove all of them in the passage and output the refined passage in a fluent and natural style, following these rules:
    1. Try to remove wrong phrases and do not use other phrases to replace
    
    Examples:
    Passage:
    In addition to the sandwiches of various sizes, a bowl, a cup, and a spoon can be seen on the table, suggesting that the guests are sharing food and drinks.
    Wrong phrases:
    ['spoon', 'drinks', 'sandwiches is various sizes']
    Refined passage: 
    In addition to the sandwiches, a bowl and a cup can be seen on the table, suggesting that the guests are sharing food.
    
    Passage:
    The image depicts a scene of two giraffes standing on a dirt road near a fence. There are three cars parked in the background, with one on the left side and two more on the right side.
    Wrong phrases:
    ['cars', 'cars are three']
    Refined passage:
    The image depicts a scene of two giraffes standing on a dirt road near a fence.

    Passage:
    {passage}
    Wrong phrases:
    {Hallu_phrase}
    Refined passage: '''
    content = PROMPT_TEMPLATE.format(passage=passage, Hallu_phrase=hallucination_phrases)
    message_example = [
        {"role": "system", "content": 'You are a language assistant that helps to refine a passage with wrong phrases removed.'},
        {"role": "user", "content": content}
    ]
    generation_data = {
        "model": 'gpt-3.5-turbo',
        "messages": message_example,
        "temperature": 0.5
    }
    results = send_request(generation_data)
    retries = 0
    while (results == None or 'error' in results) and retries < 5:
        results = send_request(generation_data)
        retries += 1
    if results is not None and 'error' not in results:
        return results
    else:
        return ""

def LLM_evaluation_hallucination(coco_captions, bounding_box, description_1, description_2, description_3, description_4, description_5):
    PROMPT_TEMPLATE='''Suppose you are a hallucination annotator who judges the degree of hallucination based on the number of errors in the description of objects, relationships, and attributes, and you have the following real image information. 
    Reference captions: {coco_captions}
    Bounding box: {bounding_box}
    Please just provide the hallucination score(1-5) for the below descriptions without any explanation, where the fewer descriptive errors in the caption, the higher the hallucination score given. The output format: [x,...]
    Descriptions:
    caption 1: {description_1}
    caption 2: {description_2}
    caption 3: {description_3}
    caption 4: {description_4}
    caption 5: {description_5}
    Output: '''

    total_prompt = PROMPT_TEMPLATE.format(coco_captions=coco_captions,bounding_box=bounding_box,description_1=description_1,description_2=description_2,description_3=description_3,description_4=description_4,description_5=description_5)
    message_example = [
        {"role": "user", "content": total_prompt}
    ]
    generation_data = {
        "model": 'gpt-3.5-turbo',
        "messages": message_example,
        "temperature": 0.8
    }
    results = send_request(generation_data)
    retries = 0
    while (results == None or 'error' in results) and retries < 5:
        results = send_request(generation_data)
        retries += 1
    if results is not None and 'error' not in results:
        return results
    else:
        return ""

def LLM_evaluation_details(coco_captions, bounding_box, description_1, description_2, description_3, description_4, description_5):
    PROMPT_TEMPLATE='''Suppose you are an image detail annotator who judges the degree of sentence diversity based on the number of objects, relations, and attributes. 
    Please just provide the diversity score(1-5) for the below descriptions without any explanation, where longer caption with more content give a higher diversity score. The output format: [x,...]
    Descriptions:
    caption 1: {description_1}
    caption 2: {description_2}
    caption 3: {description_3}
    caption 4: {description_4}
    caption 5: {description_5}
    Output: '''

    total_prompt = PROMPT_TEMPLATE.format(coco_captions=coco_captions,bounding_box=bounding_box,description_1=description_1,description_2=description_2,description_3=description_3,description_4=description_4,description_5=description_5)
    message_example = [
        {"role": "user", "content": total_prompt}
    ]
    generation_data = {
        "model": 'gpt-3.5-turbo',
        "messages": message_example,
        "temperature": 0.8
    }
    results = send_request(generation_data)
    retries = 0
    while (results == None or 'error' in results) and retries < 5:
        results = send_request(generation_data)
        retries += 1
    if results is not None and 'error' not in results:
        return results
    else:
        return ""
