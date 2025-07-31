conda activate finetune
# LoRA-Based Fine-Tuning of ESM-2
python Finetune-ESM2.py

# train LoFT-TCR
# encode
conda activate dgl_env
python run_loraESM2.py -sd iedb_McPAS_vdjdb1_2_3_5folds -td fold{fold_num} # binary
python run_loraESM2.py -sd tri_iedb_McPAS_vdjdb1_2_3_5folds -td fold{fold_num} # trinary
# train
python run_LoFT-TCR.py -m trinary -sd tri_iedb_McPAS_vdjdb1_2_3_5folds -td fold{fold_num} -cu 0 -e 1000
python run_LoFT-TCR.py -m binary -sd iedb_McPAS_vdjdb1_2_3_5folds -td fold{fold_num} -cu 0 -e 1000
# test
python test_LoFT-TCR.py -sd exp_datasets -td tri_iedb_McPAS_vdjdb1_2_3_5folds_fold{fold_num}_test -tmd tri_iedb_McPAS_vdjdb1_2_3_5folds_fold{fold_num}lora -m trinary
python test_LoFT-TCR.py -sd exp_datasets -td bi_iedb_McPAS_vdjdb1_2_3_5folds_fold{fold_num}_test -tmd iedb_McPAS_vdjdb1_2_3_5folds_fold{fold_num}lora -m binary

# Dimensionality reduction
python encodeAndUmap.py
