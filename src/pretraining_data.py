from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np

def load_and_prepare_wikitext(tokenizer_name="bert-base-uncased", 
                              block_size=128,
                              masking_probability=0.15):
    """
    Loads WikiText-2 and prepares it for Masked Language Modeling pretraining.
    Returns train_dataset, val_dataset in Hugging Face Dataset format, each with:
      - keys: input_ids, attention_mask, labels
      - shapes: [num_blocks, block_size]
    
    Masking details:
    - Randomly masks tokens with masking_probability (default 15%)
    - Labels are -100 for unmasked tokens (ignored in loss)
    - Labels contain original tokens for masked positions
    """

    # dict with splits : train, validation, test
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Ensure the tokenizer has a mask token
    if tokenizer.mask_token is None:
        raise ValueError(
            "Tokenizer does not have a mask token."
        )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            return_special_tokens_mask=True  # We'll use this to avoid masking special tokens
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"]
    )

    # 4) Group text into blocks of length `block_size`
    def group_texts(examples):
        # Concatenate all texts in the batch
        concatenated = {}
        for k in examples.keys():
            all_vals = sum(examples[k], [])  # flatten list of lists
            concatenated[k] = all_vals

        total_length = len(concatenated["input_ids"])
        # Drop the remainder if total_length < block_size
        total_length = (total_length // block_size) * block_size

        # Split by chunks of block_size
        result = {}
        for k in concatenated.keys():
            result[k] = [
                concatenated[k][i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        return result

    def create_labels(examples):
 
        examples["labels"] = []
        examples["masked_input_ids"] = []
        
        for sequence, special_tokens_mask in zip(examples["input_ids"], 
                                               examples["special_tokens_mask"]):
            labels = [-100] * len(sequence)
            masked_sequence = sequence.copy()
            
            # Only consider tokens that are not special tokens for masking
            candidate_indices = [i for i, is_special in enumerate(special_tokens_mask) 
                               if not is_special]
            
            # Calculate number of tokens to mask
            num_to_mask = max(1, int(len(candidate_indices) * masking_probability))
            
            # Randomly select positions to mask
            mask_indices = random.sample(candidate_indices, num_to_mask)
            
            for idx in mask_indices:
                # Save the original token to labels
                labels[idx] = sequence[idx]
                
                prob = random.random()
                if prob < 0.8:
                    # 80% chance to mask
                    masked_sequence[idx] = tokenizer.mask_token_id
                elif prob < 0.9:
                    # 10% chance to replace with random token
                    masked_sequence[idx] = random.randint(0, tokenizer.vocab_size - 1)
                # 10% chance: keep original token
            
            examples["labels"].append(labels)
            examples["masked_input_ids"].append(masked_sequence)
        
        return examples

    # Apply MLM masking
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=tokenized_datasets.column_names["train"]
    )

    lm_datasets = lm_datasets.map(
        create_labels,
        batched=True,
        num_proc=1,
        remove_columns=["input_ids"]
    )
    
    # Rename masked_input_ids back to input_ids
    lm_datasets = lm_datasets.rename_column("masked_input_ids", "input_ids")

    # Get final datasets
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    
    return train_dataset, val_dataset, tokenizer

if __name__ == "__main__":
    train_dataset, val_dataset, tokenizer = load_and_prepare_wikitext()
    print("Sample of training dataset : ")
    for i in range(10):
        print(train_dataset["attention_mask"][i])
        print(train_dataset["input_ids"][i])
        print(train_dataset["labels"][i])

    print(f"Number of training examples: {len(train_dataset)}")