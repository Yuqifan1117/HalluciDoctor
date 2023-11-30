import torch
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def transform_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cuda"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (num_query, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (num_query, 4)
    logits.shape[0]
    # filter output box with > box_threshold (match with caption) Language-Guided Query Selection
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]
    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    boxes_filt_text = []
    pred_phrases = []
    scores = []

    for logit, box in zip(logits_filt, boxes_filt):
        # convert ids to token (filter stop-words in captions to get tokens)
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)

        boxes_filt_text.append(box)
        if with_logits: 
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())
    # print(torch.stack(boxes_filt_text))
    return boxes_filt, pred_phrases, torch.Tensor(scores)

# if iou > 0.9, we consider they are the same box
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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def box_process(box, image_pil):
    processed_box = box.clone()
    size = image_pil.size
    H, W = size[1], size[0]
    processed_box = processed_box * torch.Tensor([W, H, W, H])
    processed_box[:2] = processed_box[:2] - processed_box[2:] / 2
    processed_box[2:] = processed_box[:2] + processed_box[2:]
    return processed_box

def show_box(box, ax, label, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def visualize_box(file, boxes, labels):
    plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    for box, label in zip(boxes, labels):
        show_box(box.numpy(), plt.gca(), label, random_color=True)
    plt.axis('off')
    plt.savefig(file, bbox_inches="tight")

    
