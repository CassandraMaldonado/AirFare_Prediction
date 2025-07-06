# BERT Embedding Comparison 
# [CLS] Token vs. Average Embedding

This project investigates different methods for representing sentence level information using BERT embeddings and their performance in classification tasks using KNN. Specifically, it compares:

- The [CLS] token embedding.
- The average of all token embeddings.

It also explores how dimensionality reduction techniques such as PCA and UMAP affect classifier performance when reducing 768 dimensional embeddings down to 2D.

## Goals

- Compare the performance of [CLS] vs. average token embeddings using a KNN classifier.
- Visualize and evaluate the impact of reducing embedding dimensionality using:
  - Principal Component Analysis (PCA).
  - Uniform Manifold Approximation and Projection (UMAP).
- Quantify performance using F1 scores.
