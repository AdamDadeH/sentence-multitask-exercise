from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import DatasetDict

def load_and_prepare_arxiv(tokenizer_name="bert-base-uncased", max_length=512):
    """
    Loads ArXiv-10 dataset and prepares it for classification.
    Returns train_dataset, val_dataset in Hugging Face Dataset format, each with:
      - keys: input_ids, attention_mask, labels
      - shapes: [num_examples, max_length]
    """
    raw_datasets = load_dataset("effectiveML/ArXiv-10")
    # Split train dataset into train and validation
    split_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=42)
    
    raw_datasets = DatasetDict({
        "train": split_datasets["train"],
        "test": split_datasets["test"]
    })
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def preprocess(examples):
        # Combine title and abstract
        texts = [f"[CLS] {title} [SEP] {abstract}" for title, abstract in 
                zip(examples['title'], examples['abstract'])]
        
        # Tokenize
        encodings = tokenizer(texts, truncation=True, padding='max_length',
                            max_length=max_length, return_tensors='pt')
        
        # Convert labels to ids
        label_map = {
            'astro-ph': 0, 
            'cond-mat': 1, 
            'cs': 2, 
            'hep-ph': 3, 
            'hep-th': 4, 
            'math': 5, 
            'eess': 7, 
            'physics': 8, 
            'stat': 9, 
            'quant-ph': 10
        }
        labels = [label_map[label] for label in examples['label']]
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }
    
    processed_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        num_proc=1,  # Increase if you have multiple CPU cores
        remove_columns=raw_datasets["train"].column_names
    )
    
    return processed_datasets["train"], processed_datasets["test"], tokenizer

def load_and_prepare_science_ie(tokenizer_name="bert-base-uncased", max_length=512):
    """
    Loads Science-IE dataset and prepares it for token classification (NER).
    Returns train_dataset, val_dataset in Hugging Face Dataset format, each with:
      - keys: input_ids, attention_mask, labels
      - shapes: [num_examples, max_length]

    Label Mapping : {"O": 0, "B-Material": 1, "I-Material": 2, "B-Process": 3, "I-Process": 4, "B-Task": 5, "I-Task": 6}
    """
    raw_datasets = load_dataset("DFKI-SLT/science_ie", "ner")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def preprocess(examples):
        # Tokenize
        encodings = tokenizer(
            examples['tokens'],
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Prepare labels (no mapping needed as they're already integers)
        labels = examples['tags']
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }
    
    processed_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        num_proc=1,  # Increase if you have multiple CPU cores
        remove_columns=raw_datasets["train"].column_names
    )
    
    return processed_datasets["train"], processed_datasets["validation"], tokenizer

# Example usage
if __name__ == "__main__":
    # Test ArXiv dataset loading
    train_dataset, test_dataset, tokenizer = load_and_prepare_arxiv()
    print("\nArXiv Dataset Example:")
    print(train_dataset[0:10])
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of test examples: {len(test_dataset)}")
    
    # Test Science-IE dataset loading
    train_dataset, val_dataset, tokenizer = load_and_prepare_science_ie()
    print("\nScience-IE Dataset Example:")
    print(train_dataset[0:10])
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")
