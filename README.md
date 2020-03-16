# Singular Value Decomposition for prevention of the Graph Convolutional Network overfitting

## Abstract
Our project is an effort to prevent the graph neural network from being overfitted. Graph convolutional
networks are usually shallow, with the number of layers not larger than 2.  Deep graph networks
perform much worse, even if some standard techniques like dropout and weight penalizing are being
implemented.  In our work, we use singular value decomposition(SVD) for extracting the most
relevant components from embeddings constructed by the network to overcome overfitting.
