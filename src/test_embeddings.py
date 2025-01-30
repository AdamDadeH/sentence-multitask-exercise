import torch
from base_transformer import BaseTransformerConfig, TransformerWithHeads, SentenceEmbeddingHead
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def test_sentence_embeddings():
    # Load the pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./base_transformer")
    
    # Create config and model
    config = BaseTransformerConfig.from_pretrained("./base_transformer")
    model = TransformerWithHeads(config)
    model = model.from_pretrained("./base_transformer")  # This will handle safetensors automatically
    model.add_head("embedding", SentenceEmbeddingHead(config))
    model.eval()

    # Sample sentences
    sentences = [
        "Cats love knocking things over.",
        "Dogs are friendly animals.",
        "Cats and dogs are both great.",
        "I'm originally from Kansas City Missouri",
        "I've lived in both Colorado and Pennsylvania",
        "I've spent a lot of time in Japan.",
        "Measure theory is the foundation of modern probability theory.",
        "The group representations forms an orthogonal basis for the space of functions.",
        "Fermat's Last Theorem is a famous theorem in number theory."
    ]

    # Tokenize sentences
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=config.max_position_embeddings,
        return_tensors="pt"
    )

    # Get embeddings
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            head="embedding"
        )
        embeddings = outputs  # Shape: [batch_size, hidden_size]

    # Print embedding shapes and similarities
    print(f"\nEmbedding shape: {embeddings.shape}")
    
    # Calculate cosine similarities between sentences
    def cosine_similarity(a, b):
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot points with different colors for each group
    colors = ['red'] * 3 + ['blue'] * 3 + ['green'] * 3
    groups = ['Animals'] * 3 +  ['Geography'] * 3 +  ['Math'] * 3
    
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=colors[i], label=groups[i])
        plt.annotate(sentences[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.title("2D PCA Projection of Sentence Embeddings")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nCosine similarities between sentences:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"\nSimilarity between:\n'{sentences[i]}'\nand\n'{sentences[j]}':\n{sim.item():.4f}")

if __name__ == "__main__":
    test_sentence_embeddings()
