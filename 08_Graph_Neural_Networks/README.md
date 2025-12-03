# Graph Neural Networks (GNN)

## Overview
**Graph Neural Networks (GNNs)** are a class of deep learning methods designed to perform inference on data described by graphs. Unlike standard neural networks that work on Euclidean data (images, text, tabular), GNNs work on non-Euclidean data where the relationships between entities (nodes) are as important as the entities themselves.

## Key Concepts
1.  **Graph Structure**: Defined by $G = (V, E)$, where $V$ is the set of nodes and $E$ is the set of edges.
2.  **Adjacency Matrix ($A$)**: A matrix representation of the graph connections.
3.  **Node Features ($X$)**: Attributes associated with each node.
4.  **Message Passing**: The core mechanism where nodes exchange information with their neighbors to update their own representation (embedding).

## Directory Structure
*   **Graph_Convolution**: Methods based on spectral or spatial convolution.
    *   `GCN_Spectral`: Spectral Graph Convolutional Networks (Kipf & Welling).
*   **Graph_Attention**: Methods that use attention mechanisms to weigh neighbors.
    *   `GAT`: Graph Attention Networks (Veličković et al.).

## Applications
*   **Social Network Analysis**: Community detection, link prediction.
*   **Drug Discovery**: Molecular property prediction.
*   **Recommendation Systems**: User-item interaction graphs.
*   **Traffic Forecasting**: Road network modeling.
