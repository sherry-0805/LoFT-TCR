import torch
import esm
import os, sys
from config import *
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, inject_adapter_in_model

#config
root = os.path.join(args.pridir, args.secdir, args.terdir)
test_csv = os.path.join(root, f'test.tsv')
test_path_pickle_cdr3 = os.path.join(root, 'esm2_feature_cdr3_test')
test_path_pickle_peptide = os.path.join(root, 'esm2_feature_peptide_test')


# Load data
print('Loading the test data..')
test_data = pd.read_csv(test_csv, delimiter='\t')

# Load ESM-2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D.pt")
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()


def encode_batch(data):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    sequence_representations = torch.stack([token_representations[i, 1:tokens_len - 1].mean(0) for i, tokens_len in enumerate(batch_lens)])
    sequence_representations = sequence_representations.to(device)  # Move to GPU if not already

    torch.cuda.empty_cache()
    return sequence_representations
def encode_batch(model, tokenizer, df, device):
    embeddings = []
    for _, seq in tqdm(df, desc="Encoding CDR3"):
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

    return np.vstack(embeddings)

def split_into_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def encode_and_save(data, cdr3_path, peptide_path):
    cdr3_data = [(f"cdr3_{i}", row["cdr3"]) for i, row in data.iterrows()]
    peptide_data = [(f"peptide_{i}", row["peptide"]) for i, row in data.iterrows()]

    cdr3_batches = split_into_batches(cdr3_data, batch_size=1000)
    peptide_batches = split_into_batches(peptide_data, batch_size=1000)

    cdr3_dict = {}
    peptide_dict = {}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True)
    base_model = AutoModelForMaskedLM.from_pretrained(args.base_model_path)
    lora_config = LoraConfig(r=4, lora_alpha=16, bias="none", target_modules=["query", "key", "value", "dense"])
    model = inject_adapter_in_model(lora_config, base_model)
    adapter_path = os.path.join("./tcr_mlm_lora_output/checkpoint-463/", "lora_adapter.pth")
    model.load_state_dict(torch.load(adapter_path, map_location="cpu"), strict=False)
    model = model.to(device).eval()
    
    for batch in cdr3_batches:
        cdr3s_encoding = encode_batch(model, tokenizer, batch, device)
        for cdr3, encoding in zip([x[1] for x in batch], cdr3s_encoding):
            cdr3_dict[cdr3] = encoding
        with open(cdr3_path, 'wb') as f:
            pickle.dump(cdr3_dict, f)

    for batch in peptide_batches:
        peptides_encoding = encode_batch(batch)
        for peptide, encoding in zip([x[1] for x in batch], peptides_encoding):
            peptide_dict[peptide] = encoding
        with open(peptide_path, 'wb') as f:
            pickle.dump(peptide_dict, f)

    print(f"ESM2 feature has saved in {cdr3_path} and {peptide_path}")

print('Encoding and saving test data features...')
encode_and_save(test_data, test_path_pickle_cdr3, test_path_pickle_peptide)