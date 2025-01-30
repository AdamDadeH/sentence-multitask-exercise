import torch
import torch.nn as nn
from base_transformer import BaseTransformerConfig, TransformerWithHeads

class SentenceClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, hidden_states, labels=None):
        # Use [CLS] token output (first token)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return loss, logits

class TokenClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, hidden_states, labels=None):
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        # Get attention mask from hidden states by checking for padding
        attention_mask = (hidden_states.sum(dim=-1) != 0).long()

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            
        return loss, logits

def create_multitask_model(config):
    # Create base model with language modeling head
    model = TransformerWithHeads.from_pretrained("./base_transformer")
    arxiv_num_labels = 10
    science_ie_num_labels = 7
    model.add_head("classification", SentenceClassificationHead(config, arxiv_num_labels))
    model.add_head("token_classification", TokenClassificationHead(config, science_ie_num_labels))
    return model


def test_model_forward():
    # Initialize config and model
    config = BaseTransformerConfig(
        vocab_size=30522,  # Standard BERT vocab size
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512
    )
    
    model = create_multitask_model(config)
    
    # Create dummy inputs
    batch_size = 2
    seq_length = 512
    dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    dummy_attention_mask = torch.ones((batch_size, seq_length))
    dummy_classification_labels = torch.randint(0, 10, (batch_size,))  # arxiv labels
    dummy_token_labels = torch.randint(0, 7, (batch_size, seq_length))  # science_ie labels
    
    # Test classification head
    classification_output = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        head="classification",
        labels=dummy_classification_labels
    )
    
    # Test token classification head
    token_classification_output = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        head="token_classification",
        labels=dummy_token_labels
    )
    
    print("Classification loss:", classification_output[0])
    print("Classification logits shape:", classification_output[1].shape)
    print("Token classification loss:", token_classification_output[0])
    print("Token classification logits shape:", token_classification_output[1].shape)

    both_output = model(input_ids = dummy_input_ids, attention_mask = dummy_attention_mask, labels = {"classification": dummy_classification_labels, "token_classification": dummy_token_labels})
    print(both_output)
    
if __name__ == "__main__":
    test_model_forward()

