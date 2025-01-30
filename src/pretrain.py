import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from base_transformer import BaseTransformerConfig, MaskedLanguageModelingHead, TransformerWithHeads
from pretraining_data import load_and_prepare_wikitext



def create_model(config):
    # Create base model with language modeling head
    model = TransformerWithHeads(config)
    model.add_head("mlm", MaskedLanguageModelingHead(config))
    return model


def pretrain(
    epochs=1,
    batch_size=2,
    lr=5e-4,
    block_size=128,
    tokenizer_name="bert-base-uncased"
):
    # 1) Prepare data
    train_dataset, val_dataset, tokenizer = load_and_prepare_wikitext(
        tokenizer_name=tokenizer_name,
        block_size=block_size
    )

    # 2) Create torch DataLoader
    def collate_fn(samples):
        # Each sample is a dict with input_ids, attention_mask, labels
        input_ids = [torch.tensor(s["input_ids"], dtype=torch.long) for s in samples]
        attention_masks = [torch.tensor(s["attention_mask"], dtype=torch.long) for s in samples]
        labels = [torch.tensor(s["labels"], dtype=torch.long) for s in samples]

        # Stack
        input_ids = torch.stack(input_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        labels = torch.stack(labels, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    config = BaseTransformerConfig(
        vocab_size=tokenizer.vocab_size,   # or tokenizer.vocab_size + (any special tokens)
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512
    )

    model = create_model(config)
    model.train()
    model = model.cuda() if torch.cuda.is_available() else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    # 5) Training loop
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask=attention_mask, labels=labels, head="mlm")[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"Step {step} - loss: {loss.item():.4f}")

        # (Optional) Evaluate on val set at epoch end
        val_loss = 0.0
        val_steps = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                val_steps += 1
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    labels = labels.cuda()

                loss = model(input_ids, attention_mask=attention_mask, labels=labels, head="mlm")["loss"]
                val_loss += loss.item()

        avg_val_loss = val_loss / max(val_steps, 1)
        print(f"Validation Loss after epoch {epoch+1}: {avg_val_loss:.4f}")

        model.train()

    print("Training complete.")

    model.save_pretrained("./base_transformer")
    tokenizer.save_pretrained("./base_transformer")
    print("Model & tokenizer saved.")

    return model, tokenizer

if __name__ == "__main__":
    trained_model, trained_tokenizer = pretrain(
        epochs=10,
        batch_size=2,
        lr=5e-4,
        block_size=128,
        tokenizer_name="bert-base-uncased"
    )

