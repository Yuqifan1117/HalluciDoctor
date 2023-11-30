import torch 
import torchvision
import numpy as np
import sng_parser
import matplotlib.pyplot as plt


def generate_sng_tags(caption):
    parser = sng_parser.Parser('spacy', model='en_core_web_sm')
    graph = parser.parse(caption)

    obj_name = [i["span"] for i in graph['entities']]
    relations = []
    for relation in graph['relations']:
        subject_name = obj_name[relation['subject']]
        object_name = obj_name[relation['object']]
        predicate = relation['relation']
        relations.append([subject_name, predicate, object_name])
    return relations, graph

def generate_textgraph(caption, tokenizer, model, device):
    text_graphs = []
    text = tokenizer(f"Generate Scene Graph: {caption}", max_length=400, return_tensors="pt", truncation=True).to(device)
    model = model.to(device)
    generated_ids = model.generate(
    text["input_ids"],
        attention_mask=text["attention_mask"],
        use_cache=True,
        decoder_start_token_id=tokenizer.pad_token_id,
        num_beams=1,
        max_length=200,
        early_stopping=True
        )
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    result = result[2:-2]
    result = result.split(' ), ( ')
    for triplet in result:
        text_graphs.append(triplet.split(', '))
    return text_graphs


def box_process(box, image_pil):
    processed_box = box.clone()
    size = image_pil.size
    H, W = size[1], size[0]
    processed_box = processed_box * torch.Tensor([W, H, W, H])
    processed_box[:2] = processed_box[:2] - processed_box[2:] / 2
    processed_box[2:] = processed_box[:2] + processed_box[2:]
    return processed_box

def IoU(b1, b2):

    if b1[2] <= b2[0] or \
        b1[3] <= b2[1] or \
        b1[0] >= b2[2] or \
        b1[1] >= b2[3]:
        return 0
    b1b2 = np.vstack([b1,b2])
    minc = np.min(b1b2, 0)
    maxc = np.max(b1b2, 0)    
    union_area = (maxc[2]-minc[0])*(maxc[3]-minc[1])
    box1Area = (b1[3]-b1[1])*(b1[2]-b1[0])
    box2Area = (b2[3]-b2[1])*(b2[2]-b2[0])
    int_area = max((minc[2]-maxc[0])*(minc[3]-maxc[1]), 0)
    return float(int_area)/float(box2Area+box1Area-int_area)

    