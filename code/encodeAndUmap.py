
import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from umap import UMAP
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, inject_adapter_in_model
import gc
from sklearn.preprocessing import StandardScaler
def perform_umap(features,
                umap_n_neighbors=15, 
                umap_min_dist=0.1, 
                umap_metric='euclidean', 
                random_state=42):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("UMAP...")
    reducer = UMAP(n_components=2, 
                            n_neighbors=umap_n_neighbors, 
                            min_dist=umap_min_dist, 
                            metric=umap_metric, 
                            random_state=random_state)
    umap_features = reducer.fit_transform(scaled_features)
    return umap_features
def encode_and_umap(model, tokenizer, df, device):
    embeddings = []
    for seq in tqdm(df["CDR3"], desc="Encoding CDR3"):
        inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=22)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[33]

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            lengths = attention_mask.sum(dim=1)

            for i in range(hidden_states.size(0)):
                valid_len = lengths[i].item()
                start = 1
                end = valid_len - 1 if valid_len > 2 else valid_len
                token_repr = hidden_states[i, start:end, :]
                mean_pooled = token_repr.mean(dim=0)
                embeddings.append(mean_pooled.cpu().numpy())

    embedding_matrix = np.vstack(embeddings)
    return perform_umap(embedding_matrix)

def process_checkpoint(checkpoint_dir, df, tokenizer, base_model_path, device):
    print(f"Processing model: {checkpoint_dir}")
    base_model = AutoModelForMaskedLM.from_pretrained(base_model_path)
    lora_config = LoraConfig(r=4, lora_alpha=args.lora_alpha, bias=args.bias, target_modules=target_modules_options[args.index])
    print(f"Model parameters: {args.lora_alpha}, {args.bias}, {target_modules_options[args.index]}")
    model = inject_adapter_in_model(lora_config, base_model)
    adapter_path = os.path.join(checkpoint_dir, "lora_adapter.pth")
    model.load_state_dict(torch.load(adapter_path, map_location="cpu"), strict=False)
    model = model.to(device).eval()

    umap_coords = encode_and_umap(model, tokenizer, df, device)
    df_out = df.copy()
    df_out["umap1"] = umap_coords[:, 0]
    df_out["umap2"] = umap_coords[:, 1]
    df_out.to_csv(os.path.join(checkpoint_dir, "embedding_umap.csv"), index=False)
    print("✅ LoRA CSV saved")
def process_base_model(df, tokenizer, base_model_path, output_dir, device):
    print("Processing base model...")
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs(output_dir, exist_ok=True)
    model = AutoModelForMaskedLM.from_pretrained(base_model_path).to(device).eval()
    umap_coords = encode_and_umap(model, tokenizer, df, device)
    df_out = df.copy()
    df_out["umap1"] = umap_coords[:, 0]
    df_out["umap2"] = umap_coords[:, 1]
    df_out.to_csv(os.path.join(output_dir, "embedding_umap.csv"), index=False)
    print("✅ Base model CSV saved")

target_modules_options = [
        ["query", "key", "value", "dense"],
        ["query", "value"],
        ["query"]
    ]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="./local_models/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c")
    parser.add_argument("--data_path", type=str, default="../data/25_07_11_filtered_vdjdb.csv")
    parser.add_argument("--checkpoint_root", type=str, default="./tcr_mlm_lora_output/")
    parser.add_argument("--bias", type=str, default="none")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--index", type=int, choices=[0, 1, 2], default=0,
                       help="Index for target_modules configuration: "
                            "0=['query', 'key', 'value', 'dense'], "
                            "1=['query', 'value'], "
                            "2=['query']")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, delimiter=",")
    df["CDR3_length"] = df["CDR3"].apply(len)
    df = df[["CDR3", "V", "J", "Epitope species", "CDR3_length"]].dropna()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_dirs = sorted([
        os.path.join(args.checkpoint_root, d) for d in os.listdir(args.checkpoint_root)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.checkpoint_root, d))
    ])
    for ckpt_dir in all_dirs:
        process_checkpoint(ckpt_dir, df, tokenizer, args.base_model_path, device)


    base_output_dir = os.path.join(args.checkpoint_root, "base_model")
    process_base_model(df, tokenizer, args.base_model_path, base_output_dir, device)
