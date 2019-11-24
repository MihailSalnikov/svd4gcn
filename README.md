# SVD for reducing overfitting on GCN

## Plan:
1. Make neural network with GCN with big amount of params as a embedding on some popular dataset, train it and observ overfitting (because big amount of params)
2. Use SVD for decompose embedding, trancate it by **r**
3. Fit new version of model with truncated embedding matrixes and check if this has affect overfittin. We expect, that if we choose good **r**, it can reduce overfittin

## Materials

* Zachary Karate club [description](https://en.wikipedia.org/wiki/Zachary%27s_karate_club)
* [Zachary Karate club dataset](https://networkx.github.io/documentation/stable/auto_examples/graph/plot_karate_club.html) from NetworkX
* Good [blogpost](https://tkipf.github.io/graph-convolutional-networks/#footer) about GCN
* [Paper](http://web.cs.ucla.edu/~yzsun/papers/2019_WSDM_SimGNN.pdf) with example of using GCN as a embedding
* [Example](https://github.com/AngusMonroe/KarateGCN/tree/master) of GCN on PyTorch on Karate club dataset