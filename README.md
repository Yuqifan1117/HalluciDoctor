# HalluciDoctor: Mitigating Hallucinatory Toxicity in Visual Instruction Data
![example-2_00](https://github.com/Yuqifan1117/HalluciDoctor/assets/48062034/659bedd2-5da0-42dc-bf60-230f6aa03445)
## 🔥 News
- This paper is accepted by CVPR 2024 (2024.2.27)
- This is the official repository for the paper "HalluciDoctor: Mitigating Hallucinatory Toxicity in Visual Instruction Data". 🍇 [[Read our arXiv Paper](https://arxiv.org/abs/2311.13614)].
- We update the main code of **HalluciDoctor** and corresponding datasets *LLaVA+* and *LLaVA++*.


## ⭐ Steps
- Dataset preparation: LLaVA-158K; coco_category.json; coco_object_co_occur.json; object_sample for 'counterfactual images' following **[INSTALL.md](INSTALL.md)**.
- HalluciDoctor Framework
    1. PYTHONPATH=./ python models/question_generator.py
    2. PYTHONPATH=./ python models/blip2_candidate_answer_generator.py; PYTHONPATH=./ python models/instructblip_candidate_answer_generator.py; PYTHONPATH=./ python models/minigpt4_candidate_answer_generator.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    3. PYTHONPATH=./ python models/consistency_crosscheck.py; PYTHONPATH=./ python models/consistency_crosscheck_object.py
    4. PYTHONPATH=./ python models/refine_dataset.py -> **LLaVA+**
    5. PYTHONPATH=./ python models/seesaw_counterfactual_generation.py -> **LLaVA++**
- MLLM fine-tuning on **LLaVA+** and **LLaVA++** in the refined_datasets.

## 📜 Citation
If you find this work useful for your research, please cite our paper and star our git repo:
```bibtex
@misc{yu2023hallucidoctor,
      title={HalluciDoctor: Mitigating Hallucinatory Toxicity in Visual Instruction Data}, 
      author={Qifan Yu and Juncheng Li and Longhui Wei and Liang Pang and Wentao Ye and Bosheng Qin and Siliang Tang and Qi Tian and Yueting Zhuang},
      year={2023},
      eprint={2311.13614},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
