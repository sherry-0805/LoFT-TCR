import os
import torch
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, inject_adapter_in_model
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bias", type=str, default="none")
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--output_dir", type=str, default="./tcr_mlm_lora_output/")
parser.add_argument("--port", type=str, default="9993")
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--index", type=int, choices=[0, 1, 2], default=0,
                       help="Index for target_modules configuration: "
                            "0=['query', 'key', 'value', 'dense'], "
                            "1=['query', 'value'], "
                            "2=['query']")
parser.add_argument("--dataset", type=int, choices=[0, 1], default=0,
                       help="Index for dataset: 0=vdjdb, 1=IEDB_McPAS_VDJDB")
args_parser = parser.parse_args()
# ------------------------------------
# Configurable target_modules options
# ------------------------------------
target_modules_options = [
        ["query", "key", "value", "dense"],
        ["query", "value"],
        ["query"]
    ]
# ----------------------
# Environment Settings
# ----------------------
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = args_parser.port
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
print("Torch version: ",torch.__version__)
print("Cuda version: ",torch.version.cuda)
print("Numpy version: ",np.__version__)
print("Pandas version: ",pd.__version__)

# ----------------------
# DeepSpeed Settings
# ----------------------
ds_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

# -----------------------------------
# Set seed for reproducible results.
# -----------------------------------
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

# ----------------------
# TCR Data Processing & Loading
# ----------------------
def load_tcr_data(path):
    df = pd.read_csv(path)
    print(f"✅ TCR dataset loaded: {path}, total {len(df)} records")

    if args_parser.dataset == 0:
        train_df, valid_df, test_df = np.split(
            df.sample(frac=1, random_state=42),
            [int(0.8 * len(df)), int(0.9 * len(df))]
        )
    else :
        train_df = df
        valid_size = int(0.1 * len(train_df))
        test_size = int(0.1 * len(train_df)) 
        train_df = train_df.sample(frac=1, random_state=42)
        valid_df = train_df.iloc[:valid_size]
        test_df = train_df.iloc[valid_size:valid_size + test_size]
        train_df = train_df.iloc[valid_size + test_size:]

    return train_df, valid_df, test_df


# -----------------------------------
# Data Processing - Creating Dataset
# -----------------------------------
def create_dataset(tokenizer, sequences, max_length=22):
    tokenized = tokenizer(sequences, padding="max_length", truncation=True, max_length=max_length)
    return Dataset.from_dict(tokenized)

# ---------------------------------------------------------------
# Loading Masked Language Model with Low-Rank Adaptation (LoRA)
# ---------------------------------------------------------------
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def load_lora_mlm_model(checkpoint_path, mixed=True):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16 if mixed else None)
    print(f"✅Trainable parameters before LoRA injection:{count_trainable_params(model):,}")
    # Integrate LoRA adapters
    peft_config = LoraConfig(
        r=4,
        lora_alpha=args_parser.lora_alpha,
        bias=args_parser.bias,
        target_modules=target_modules_options[args_parser.index]
    )
    model = inject_adapter_in_model(peft_config, model)

    # Freeze all parameters except LoRA and lm_head
    for name, param in model.named_parameters():
        if "lora" not in name and "lm_head" not in name:
            param.requires_grad = False
    print(f"✅Trainable parameters after LoRA injection:{count_trainable_params(model):,}")
    
    return model, tokenizer

# ----------------------
# Evaluation Metrics
# ----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    masked_preds = predictions[mask]
    masked_labels = labels[mask]
    acc = accuracy_score(masked_labels, masked_preds)
    return {"accuracy": acc}

# ----------------------
# Main training function
# ----------------------
class StepEvalLoggerCallback(TrainerCallback):
    def __init__(self, log_list, eval_dataset, train_dataset, test_dataset, eval_steps):
        self.log_list = log_list
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_steps = eval_steps
        self.trainer = None
        self.has_logged_eval_at_step0 = False

    def set_trainer(self, trainer):
        self.trainer = trainer
    
    def on_step_begin(self, args, state, control, **kwargs):
        # Pre-training evaluation at step 0 (baseline metrics)
        if not self.has_logged_eval_at_step0 and state.global_step == 0 and self.trainer is not None:
            print("📊 Step 0 Initial Model Evaluation...")
            if self.train_dataset is not None:
                train_metrics = self.trainer.evaluate(
                    eval_dataset=self.train_dataset,
                    metric_key_prefix="train"
                )
                print(f"✅ Train @ step 0: {train_metrics}")

            if self.eval_dataset is not None:
                eval_metrics = self.trainer.evaluate(
                    eval_dataset=self.eval_dataset,
                    metric_key_prefix="eval"
                )
                print(f"✅ Eval @ step 0: {eval_metrics}")

            if self.test_dataset is not None:
                test_metrics = self.trainer.evaluate(
                    eval_dataset=self.test_dataset,
                    metric_key_prefix="test"
                )
                print(f"✅ Test @ step 0: {test_metrics}")

            self.has_logged_eval_at_step0 = True
    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            print("⚠️ self.trainer is None, skipping evaluation.")
            return control
        # Print results every args_parser.step steps
        if state.global_step % args_parser.step == 0 and state.global_step > 0:
            print(f"📊 Evaluating at step {state.global_step}...")

            train_metrics = self.trainer.evaluate(
                eval_dataset=self.train_dataset,
                metric_key_prefix="train"
            )
            eval_metrics = self.trainer.evaluate(
                eval_dataset=self.eval_dataset,
                metric_key_prefix="eval"
            )

            row = {
                "step": state.global_step,
                "train_loss": train_metrics.get("train_loss"),
                "train_acc": train_metrics.get("train_accuracy"),
                "eval_loss": eval_metrics.get("eval_loss"),
                "eval_acc": eval_metrics.get("eval_accuracy"),
            }
            self.log_list.append(row)
            print(f"✅ Step {state.global_step} logged: {row}")

        return control
def apply_fixed_mask(dataset, tokenizer, mlm_probability=0.15):
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    all_features = [dataset[i] for i in range(len(dataset))]
    masked = collator(all_features)

    masked_dataset = Dataset.from_dict({
        "input_ids": masked["input_ids"],
        "attention_mask": masked["attention_mask"],
        "labels": masked["labels"]
    })
    return masked_dataset

def save_lora_parameters(model, path):
    non_frozen_params = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    torch.save(non_frozen_params, path)
def load_lora_parameters(model, path):
    saved_state = torch.load(path, map_location="cpu")
    for name, param in model.named_parameters():
        if name in saved_state:
            param.data.copy_(saved_state[name].data)

def train_tcr_mlm(
    checkpoint,
    tcr_csv_path,
    output_dir,
    batch_size=8,
    accum=2,
    epochs=3,
    lr=3e-4,
    seed=42,
    deepspeed=True,
    mixed=True
):
    # Set random seed
    set_seeds(seed)
    # Load and split data
    train_df, valid_df, test_df = load_tcr_data(tcr_csv_path)
    # Load model and tokenizer
    model, tokenizer = load_lora_mlm_model(checkpoint, mixed)

    # Dataset construction
    train_dataset = create_dataset(tokenizer, list(train_df['sequence']))
    valid_dataset = create_dataset(tokenizer, list(valid_df['sequence']))
    test_dataset = create_dataset(tokenizer, list(test_df['sequence']))
    print(f"Train_df size: {len(train_dataset)}, Valid_df size: {len(valid_dataset)}, test_df size: {len(test_dataset)}")
    print(train_dataset[0])
    print(valid_dataset[0])
    print(test_dataset[0])
    print("Dataset construction completed✅")

    # Add mask with Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Estimate steps per epoch (number of steps per epoch)
    steps_per_epoch = max(1, len(train_dataset) // (batch_size * accum))

    masked_test_dataset = apply_fixed_mask(test_dataset, tokenizer, mlm_probability=0.15)# 测试集固定 mask，保证评估结果可重复性，可以和其他模型比较
    all_logs = []
    log_callback = StepEvalLoggerCallback(
        log_list=all_logs,
        eval_dataset=valid_dataset,
        train_dataset=train_dataset,
        test_dataset=masked_test_dataset,
        eval_steps=steps_per_epoch
    )

    # TrainingArguments
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=steps_per_epoch,
        save_strategy="steps",
        save_steps=steps_per_epoch,
        #save_total_limit=3,
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        deepspeed=ds_config if deepspeed else None,
        fp16=mixed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[]
    )

    log_callback.set_trainer(trainer)
    trainer.add_callback(log_callback)

    trainer.train()


    # ----------------------
    # Save model and logs
    # ----------------------
    # 1. Save LoRA fine-tuning parameters
    final_model_path = os.path.join(output_dir, "lora_adapter.pth")
    save_lora_parameters(trainer.model, final_model_path)
    # 2. Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"✅ LoRA fine-tuning parameters has saved to:{final_model_path}")
    print(f"✅ tokenizer has saved to: {output_dir}")
    
    # 3. Save each epoch model
    for epoch in range(1, epochs+1):
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{epoch*steps_per_epoch}")
        epoch_model_path = os.path.join(checkpoint_path, "lora_adapter.pth")
        if os.path.exists(checkpoint_path):
            epoch_model = AutoModelForMaskedLM.from_pretrained(checkpoint)
            peft_config = LoraConfig(
                r=4,
                lora_alpha=args_parser.lora_alpha,
                bias=args_parser.bias,
                target_modules=target_modules_options[args_parser.index]
            )
            epoch_model = inject_adapter_in_model(peft_config, epoch_model)
            
            checkpoint_state = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
            epoch_model.load_state_dict(checkpoint_state)
            
            save_lora_parameters(epoch_model, epoch_model_path)
            print(f"✅LoRA fine-tuning parameters for epoch={epoch} has saved to: {epoch_model_path}")
    # 3.Save the training log
    log_df = pd.DataFrame(all_logs)
    log_csv_path = os.path.join(output_dir, "train_step_log.csv")
    log_df.to_csv(log_csv_path, index=False)
    print(f"\n✅ Saved training logs for each step to: {log_csv_path}")

    if args_parser.dataset == 0:
        # ----------------------
        # Test Set Evaluation
        # ----------------------
        print("\nStarting evaluation on test set (Masked prediction)...")
        base_model = AutoModelForMaskedLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if mixed else None
        )
        peft_config = LoraConfig(
            r=4,
            lora_alpha=args_parser.lora_alpha,
            bias=args_parser.bias,
            target_modules=target_modules_options[args_parser.index]
        )
        best_model = inject_adapter_in_model(peft_config, base_model)
        load_lora_parameters(best_model, os.path.join(output_dir, "lora_adapter.pth"))
        
        best_model.eval()
        best_model.to("cuda" if torch.cuda.is_available() else "cpu")
        if mixed:
            best_model = best_model.half()
        

        masked_test_loader = DataLoader(
            masked_test_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda x: {
                "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in x]),
                "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in x]),
                "labels": torch.stack([torch.tensor(f["labels"]) for f in x]),
            }
        )

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(masked_test_loader):
                input_ids = batch["input_ids"].to(best_model.device)
                attention_mask = batch["attention_mask"].to(best_model.device)
                labels = batch["labels"].to(best_model.device)

                if mixed:
                    attention_mask = attention_mask.half()
                outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                mask = labels != -100
                all_preds.extend(predictions[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
        test_acc = accuracy_score(all_labels, all_preds)
        print(f"\n✅ Test set fixed masked token accuracy: {test_acc:.4f}")
        test_result = {"masked_accuracy": test_acc}

        return trainer.state.log_history, test_result
    
    return trainer.state.log_history


if __name__ == "__main__":
    checkpoint = "./local_models/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
    if args_parser.dataset == 0:
        tcr_csv_path = "../data/25_07_11_filtered_vdjdb.csv" 
    else:
        tcr_csv_path = "../data/tcr.csv" 

    history = train_tcr_mlm(
        checkpoint=checkpoint,
        tcr_csv_path=tcr_csv_path,
        output_dir=args_parser.output_dir,
        batch_size=args_parser.bs,
        accum=2,
        epochs=args_parser.epochs,
        lr=args_parser.lr,
        seed=42,
        deepspeed=True,
        mixed=True
    )

    print("\n✅ Training Done")