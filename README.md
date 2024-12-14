# Temporal-Multimodal Consistency Alignment for Alzheimzer's Cognitive Assessment Prediction

## 🚀 Library Installation

For manual installation, please run the following installation:
```
pip install -r requirements.txt
```

## 🔥 Pre-train:

* Modify the path in config file configs/pretrain_config.yaml, and ```python pretrain.py``` to pre-train.

* Run `CUDA_VISIBLE_DEVICES=0 python pretrain.py --gpus 1 --num_workers 8 --learning_rate 2e-4 --batch_size 8  --cfg_path 'configs/pretrain_config.yaml' `


## 🌟 Quick Start:

* **Data needed:**
- ADNI1GO images, need to be registered
- diagnosis clinical data: DXSUM_PDXCONV_ADNIALL_24Sep2023, ANDI1GO_postprocess_10_09_2023.csv, ADNIMERGE_05Jan2024.csv
- Gene data: ADNI_1_GWAS_Plink/ADNI_cluster_01_forward_757LONI, PLINK_APOE/extracted_snps

* **Finetuning:**

Run `CUDA_VISIBLE_DEVICES=0 python finetune.py --gpus 1 --num_workers 8 --learning_rate 5e-5 --batch_size 8 --cfg_path 'finetune_config.yaml' --weight_path 'Pretrained_weights/epoch=29-step=3450.ckpt' `