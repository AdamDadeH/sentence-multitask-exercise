import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
)
from transformers import PreTrainedModel


class BaseTransformerConfig(PretrainedConfig):
    """
    Base configuration for the transformer model
    """

    model_type = "base_transformer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings


class TransformerBase(PreTrainedModel):
    """
    Base transformer model without any heads
    
    The model is initialized with a config object that contains the model's hyperparameters.

    The model output is the final hidden states of the transformer encoder.
    """

    config_class = BaseTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Embeddings: token + positional
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Single Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            activation='gelu',
            batch_first=True
        )

        # Multiple Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers
        )

        self.post_init()   


    def get_encoder_output(self, input_ids, attention_mask=None):
        """Get transformer encoder outputs without any head processing"""

        # Get input shapes
        batch_size, seq_len = input_ids.shape

        # Token + positional embeddings
        token_embeds = self.token_embeddings(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(positions)
        hidden_states = token_embeds + position_embeds

        # Handle attention mask
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Get encoder output
        return self.transformer_encoder(
            hidden_states,
            src_key_padding_mask=src_key_padding_mask
        ) 


class TransformerWithHeads(PreTrainedModel):
    """Means to compose base transformer with multiple heads"""
    config_class = BaseTransformerConfig

    def __init__(self, config, heads=None):
        super().__init__(config)
        self.base = TransformerBase(config)
        self.heads = nn.ModuleDict({}) if heads is None else nn.ModuleDict(heads)
        self.post_init()

    def add_head(self, name, head):
        """Add a new head to the model"""
        self.heads[name] = head

    def forward(self, input_ids, attention_mask=None, head=None, labels=None):
        """
        Forward pass with optional head selection
        Args:
            head: str or None. If None, returns outputs from all heads
        """
        encoder_output = self.base.get_encoder_output(input_ids, attention_mask)

        if head is not None:
            # Use specific head
            if head not in self.heads:
                raise ValueError(f"Head {head} not found in model heads")
            return self.heads[head](encoder_output, labels)
        
        # Return outputs from all heads
        outputs = {}
        for head_name, head_module in self.heads.items():
            head_loss, head_output = head_module(encoder_output, labels=labels[head_name], attention_mask=attention_mask)
            outputs[head_name] = {"loss": head_loss, "output": head_output}
        
        return outputs


class MaskedLanguageModelingHead(nn.Module):
    """Masked language modeling head for predicting masked tokens"""

    def __init__(self, config):
        super().__init__()
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.config = config

    def forward(self, encoder_output, labels=None, attention_mask=None):
        """
        Args:
            encoder_output: [batch_size, seq_len, hidden_size]
            labels: [batch_size, seq_len] with -100 for non-masked tokens
        """
        logits = self.mlm_head(encoder_output)  # [batch_size, seq_len, vocab_size]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
        
        return loss, logits


class SentenceEmbeddingHead(nn.Module):
    """Head that does mean pooling over the last hidden state to return sentence embedding"""
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, encoder_output, labels=None, attention_mask=None):
        """
        Args:
            encoder_output: [batch_size, seq_len, hidden_size]
        """
        return encoder_output.mean(dim=1)






    

