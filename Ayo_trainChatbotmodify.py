from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_scheduler, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import math
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from torch.optim import AdamW
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


torch.backends.cuda.matmul.allow_tf32 = True

@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_result = []
        for feature in features:
            features_result.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
            )

        padded_result = self.tokenizer.pad(
            features_result,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {

            "input_ids": padded_result["input_ids"],
            "attention_mask": padded_result["attention_mask"]
        }

        return batch


def train():

    base_model = "tiiuae/falcon-rw-1b"
    dataset_name = "ayoolaolafenwa/sft-data"
    #dataset_name = "sam-mosaic/ift_hhrlhf_flan"
    #base_model = "gpt2"
    per_device_batch_size = 8
    max_length = 2048
    num_workers = 24 # adjust according to number of CPU cores on your machine
    learning_rate = 9.65e-6
    lr_scheduler_type = "cosine"
    num_training_epochs = 1
    train_logging_interval = 10
    model_save_interval = 1000
    enable_gradient_checkpointing = True

    MODEL_SAVE_DIR = "Falcon1BNewTrainedModels"
    MODEL_SAVE_VALDIR = "Falcon1BNewValModels"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)

    # enable gradient checkpointing to reduce memory usage
    if enable_gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    print_trainable_parameters(model)

    def batch_preprocess(samples):

        new_samples = {
            "input_ids": [],
            "attention_mask": []
        }

        for prompt, result in zip(samples["prompt"], samples["response"]):

            sequence = prompt + result

            tokenized = tokenizer(sequence, truncation=True, padding=True, max_length=max_length)

            new_samples["input_ids"].append(tokenized["input_ids"])
            new_samples["attention_mask"].append(tokenized["attention_mask"])

        return new_samples

    access_token = "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"
    raw_datasets = load_dataset(dataset_name, use_auth_token = access_token)
    train_data = raw_datasets["train"]
    val_data = raw_datasets["test"]

    train_data = train_data.map(batch_preprocess, batched=True, remove_columns=["prompt", "response"])
    val_data = val_data.map(batch_preprocess, batched=True, remove_columns=["prompt", "response"])

    train_dataloader = DataLoader(
        train_data,
        batch_size=per_device_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=max_length)
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=max_length)
    )

    #accelerator = Accelerator(mixed_precision="bf16")
    accelerator = Accelerator()
    
    model = accelerator.prepare(model)
    weight_decay = 0.05
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
    optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate, betas=(0.9, 0.95))

    num_training_steps = (len(train_dataloader) // per_device_batch_size) * num_training_epochs

    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, val_dataloader, lr_scheduler)

    def process_batch(batch):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)

        return outputs.loss

    for epoch in range(num_training_epochs):

        accelerator.print("Epoch: {}".format(epoch))

        model.train()

        train_loss = 0
        train_loss_len = 0

        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            loss = process_batch(batch)

            accelerator.backward(loss)

            optimizer.step()

            lr_scheduler.step()

            optimizer.zero_grad()

            train_loss += loss.float()
            train_loss_len += 1

            if i % train_logging_interval == 0 and i > 0:
                avg_loss = train_loss / train_loss_len
                perplexity = torch.exp(avg_loss)

                accelerator.print("Epoch: {}, Step: {}, Loss: {}, Perplexity: {}".format(epoch, i, avg_loss, perplexity))

            if i % model_save_interval == 0 and i > 0:

                current_global_step = epoch * len(train_dataloader) + i

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    f"{MODEL_SAVE_DIR}_step_{current_global_step}",
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                    ) 

                tokenizer.save_pretrained(f"{MODEL_SAVE_DIR}_step_{current_global_step}")

        val_loss = 0
        val_loss_len = 0
        model.eval()

        # Run validation after every epoch

        accelerator.print("Running Validation...")
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                loss = process_batch(batch)
                val_loss += loss.float()
                val_loss_len += 1

            avg_val_loss = torch.tensor(val_loss / val_loss_len)

            # aggregate loss across all gpus
            avg_val_loss = accelerator.gather(avg_val_loss).mean()

            perplexity = torch.exp(avg_val_loss)

            accelerator.print("Validation Loss: {}, Perplexity: {}".format(avg_val_loss.float(), perplexity.float()))

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f"{MODEL_SAVE_VALDIR}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                )
            
            tokenizer.save_pretrained(MODEL_SAVE_VALDIR)

if __name__ == "__main__":
    train()