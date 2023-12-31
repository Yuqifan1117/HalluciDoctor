# Dependences
## Pre-trained Models
- MLLM experts
    1. BLIP2: Salesforce/blip2-flan-t5-xl # default download
    2. InstructBLIP: blip2_vicuna_instruct # default download
        ```bash
        git clone https://github.com/salesforce/LAVIS.git
        cd LAVIS
        pip install -e .
        ```
        update *llm_model* in LAVIS/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml to the path of vicuna (vicuna-7b-1.1)
    3. MiniGPT4: https://github.com/Vision-CAIR/MiniGPT-4
- utils
    1. Answer chunks extraction: lizhuang144/flan-t5-base-VG-factual-sg (Huggingface, will be download automatically)
    2. BEM evaluation: https://huggingface.co/google/bert_uncased_L-12_H-768_A-12/blob/main/vocab.txt into **_VOCAB_PATH** in models/evaluation_utils.py; https://tfhub.dev/google/answer_equivalence/bem/1 into **_MODEL_PATH** in models/evaluation_utils.py
    3. GroundingDINO: replace the *GroundingDINO PATH* following the official setting (https://github.com/IDEA-Research/GroundingDINO)
    4. spacy & en_core_web_sm (python -m spacy download en_core_web_sm)
- datasets
    1. coco2017_train/coco2017_val into datasets/coco
    2. replace */home/qifan/datasets/coco/object_sample* with your path, where save the images of coco_objects (generated by Stable Diffusion v1.5 with corresponding prompt words)
