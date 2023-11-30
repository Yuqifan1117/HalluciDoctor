import os
from utils.utils import IoU
from utils.cropimage2image import cropimage2image
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings("ignore")
import random
import torchvision
from utils.grounded_sam import box_process, get_grounding_output, load_image, load_model, transform_image
coco_category = json.load(open('utils/coco_category.json'))
valid_nouns = []
for k in coco_category:
    valid_nouns.append(coco_category[k])

if __name__ == '__main__':
    refined_llava_details = json.load(open('results/refined_LLaVA_cap.json'))
    expand_refined_llava_details = {"annotations": []}
    for data in refined_llava_details["annotations"]:
        expand_refined_llava_details["annotations"].append(data)
    phrase_evaluation_results = json.load(open('results/bertmatching_evaluation.json'))
    object_evaluation_results = json.load(open('results/object_bertmatching_evaluation.json'))
    LLaVA_details = json.load(open('datasets/detail_23k.json'))

    # co-occurrence prior information
    obj_co_occur = json.load(open('utils/coco_object_co_occur.json'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py'
    grounded_checkpoint = 'checkpoints/groundingdino_swinb_cogcoor.pth'
    detect_model = load_model(config_file, grounded_checkpoint, device=device)
    instruction_dict = dict()
    original_response_dict = dict()

    total_id = []
    for data in tqdm(refined_llava_details['annotations']):
        original_response_dict[data['image_id']] = data['caption']
        total_id.append(data['image_id'])

    expansion_data = []
    for data in tqdm(refined_llava_details['annotations']):
        image_id = data['image_id']
        descriptions = data['caption']
        if image_id in object_evaluation_results:
            hallucination_objects = object_evaluation_results[image_id]['hallucination_objects']
        else:
            hallucination_objects = []
        if len(hallucination_objects) == 0:
            continue
        for hallucination_phrase in hallucination_objects:
            # find the most related object to cause hallucinations
            hallucination_occur = dict()
            for object in obj_co_occur:
                if object in hallucination_phrase:
                    hallucination_occur[object] = dict()
                    for occur_object in obj_co_occur[object]:
                        idx = original_response.find(occur_object)
                        if idx!=-1 and original_response[idx-1] == ' ' and occur_object != object:
                            hallucination_occur[object][occur_object] = obj_co_occur[object][occur_object]
            n_i = -1
            if len(hallucination_occur) == 1:
                for hallucination_item in hallucination_occur:
                    if len(sorted(hallucination_occur[hallucination_item].items(), key=lambda x:x[1], reverse=True)) == 0:
                        continue
                    most_hallucination_object = sorted(hallucination_occur[hallucination_item].items(), key=lambda x:x[1], reverse=True)[0][0]
                    n_i = sorted(hallucination_occur[hallucination_item].items(), key=lambda x:x[1], reverse=True)[0][1]
                    m_i = n_i
                    most_hallucination = hallucination_item
            if n_i == -1 or n_i == 0:
                continue
            augmented_facor = dict()
            for obj in obj_co_occur[most_hallucination]:
                if obj_co_occur[most_hallucination][obj] <= n_i:
                    augmented_facor[obj] = pow((n_i/max(obj_co_occur[most_hallucination][obj],1)),1)
                else:
                    augmented_facor[obj] = 1
            
            inhibiting_factor = dict()
            for obj in obj_co_occur[most_hallucination_object]:
                if obj_co_occur[most_hallucination_object][obj] <= m_i:
                    inhibiting_factor[obj] = pow((obj_co_occur[most_hallucination_object][obj]/m_i),1)
                else:
                    inhibiting_factor[obj] = 1
            sessaw_score = dict()
            for obj in obj_co_occur[most_hallucination]:
                sessaw_score[obj] = augmented_facor[obj]*inhibiting_factor[obj]
            for item in sorted(sessaw_score.items(), key=lambda x:x[1], reverse=True):
                if item[0] != most_hallucination and item[0] != most_hallucination_object:
                    expansion_sence = item[0]
                    break
            target_id = None
            retry = 0
            random.shuffle(total_id)
            for cur_image_id in total_id:
                original_response = original_response_dict[cur_image_id]
                idx3 = original_response.find(expansion_sence)
                cur_image_path = '/home/qifan/datasets/coco/train2017/{}.jpg'.format(cur_image_id)
                image_pil, image = load_image(cur_image_path)
                boxes_filt, pred_phrases, pred_scores = get_grounding_output(
                    detect_model, image, expansion_sence, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=device
                )
                retry += 1
                if len(pred_phrases) > 0:
                    target_id = cur_image_id
                    break
                if retry > 100:
                    break
            if target_id != None and [image_id, most_hallucination, target_id] not in expansion_data:
                expansion_data.append([image_id, most_hallucination, target_id])
    print(len(expansion_data))
    json.dump(expansion_data, open('results/coco_image_visual_instruct_expansion.json', 'w'))

    # image-level instruction expansion
    valid_objects = ['bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    total_sample = 0
    for data in tqdm(expansion_data):
        image_id = data[0]
        hallucination = data[1]
        target_scene_id = data[2]
        descriptions = original_response_dict[target_scene_id]
        if hallucination not in valid_objects:  # only generate common objects for counterfactual images
            continue
        target_item_path = '/home/qifan/datasets/coco/object_sample/{}'.format(hallucination)
        valid_target_names = []
        for image in os.listdir(target_item_path):
            if '_mask' in image:
                valid_target_names.append(image.split('._')[0])
        if len(valid_target_names) == 0:
            continue
        target_name = random.choice(valid_target_names)
        target_image = target_item_path + '/' + '{}.png'.format(target_name)
        target_image_mask = target_item_path + '/' + '{}._mask.png'.format(target_name)
        while not os.path.exists(target_image) or not os.path.exists(target_image_mask):
            target_name = random.choice(valid_target_names)
            target_image = target_item_path + '/' + '{}.png'.format(target_name)
            target_image_mask = target_item_path + '/' + '{}._mask.png'.format(target_name)
        mask = Image.open(target_image_mask)
        mask_array = np.array(mask)
        coords = np.argwhere(mask_array > 0)
        min_y, min_x = np.min(coords, axis=0)
        max_y, max_x = np.max(coords, axis=0)
        original_size = [max_x-min_x, max_y-min_y]

        # expansion on image
        exist_objects = object_evaluation_results[image_id]['detected_objects']
        background_image_path = '/home/qifan/datasets/coco/train2017/{}.jpg'.format(target_scene_id)
        
        image_pil, image = load_image(background_image_path)
        valid_boxes = []
        valid_phrases = []
        valid_scores = []
        for obj in exist_objects:
            boxes_filt, pred_phrases, pred_scores = get_grounding_output(
                detect_model, image, obj, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=device
            )
            for i in range(boxes_filt.shape[0]):
                valid_boxes.append(box_process(boxes_filt[i], image_pil)) # process boxes into image size (xyxy)
                valid_phrases.append(pred_phrases[i])
                valid_scores.append(pred_scores[i])
        if len(valid_boxes) > 0:
            valid_boxes = torch.stack(valid_boxes)
            valid_scores = torch.stack(valid_scores)
            nms_idx = torchvision.ops.nms(valid_boxes, valid_scores, iou_threshold=0.5).numpy().tolist()
            valid_boxes = valid_boxes[nms_idx]
            valid_phrases = [valid_phrases[idx] for idx in nms_idx]
        if original_size[0] > image_pil.size[0] or original_size[1] > image_pil.size[0]:
            scale = 0.15
        else:
            scale = 0.25
        for i in range(10):
            is_overlap = False
            is_paste = False
            if 1+int(original_size[0]*scale) >= image_pil.size[0] or 1+int(original_size[1]*scale) >= image_pil.size[1]:
                scale = scale * 0.5
                continue
            x = random.randint(1, image_pil.size[0]-1-int(original_size[0]*scale))
            y = random.randint(1, image_pil.size[1]-1-int(original_size[1]*scale))
            new_box = [x, y, x+original_size[0]*scale, y+original_size[1]*scale]
            for exist_box in valid_boxes:
                if IoU(exist_box, new_box) > 0:
                    is_overlap = True
            if not is_overlap:
                is_paste = True
                break
        if is_paste:
            result_image_pil = cropimage2image(target_image, target_image_mask, background_image_path, x, y, scale)
            result_image = transform_image(result_image_pil)
            boxes_filt, pred_phrases, pred_scores = get_grounding_output(
                detect_model, image, hallucination, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=device
            )
            if len(pred_phrases) > 0:
                result_image_pil.save(f'results/counterfactual_samples_all/{hallucination}_{target_scene_id}_{total_sample}.jpg')
                split_sentences = descriptions.split('. ')
                update_sentence = []
                for sentence in split_sentences[:-1]:
                    update_sentence.append(sentence)
                if hallucination[0] not in ["a", "e", "i", "o", "u"]:
                    new_sentence = f"There is also a {hallucination} in the image"
                else:
                    new_sentence = f"There is also an {hallucination} in the image"
                update_sentence.append(new_sentence)
                update_sentence.append(split_sentences[-1])
                update_descriptions = '. '.join(update_sentence)
                expand_refined_llava_details['annotations'].append({
                    "image_id": f'{hallucination}_{target_scene_id}_{total_sample}',
                    "caption": update_descriptions
                })
                total_sample += 1
    print(len(expand_refined_llava_details['annotations']))
    json.dump(expand_refined_llava_details, open('results/expand_refined_LLaVA_cap.json', 'w'))

    
