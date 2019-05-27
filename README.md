# TAFE-Net

This is the PyTorch Implementation for our paper: 

TAFE-Net: Task-aware Feature Embeddings for Low Shot Learning (CVPR 2019)

[Xin Wang](https://people.eecs.berkeley.edu/~xinw/), [Fisher Yu](https://www.yf.io/), Ruth Wang, [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Joseph E. Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/)

ArXiv link: [https://arxiv.org/abs/1904.05967](https://arxiv.org/abs/1904.05967)

**Abstract** 

Learning good feature embeddings for images often requires substantial training data. As a consequence, 
in settings where training data is limited (e.g., few-shot and zero-shot learning), we are typically forced 
to use a generic feature embedding across various tasks. Ideally, we want to construct feature embeddings 
that are tuned for the given task. In this work, we propose Task-Aware Feature Embedding Networks (TAFE-Nets) 
to learn how to adapt the image representation to a new task in a meta learning fashion. Our network is composed 
of a meta learner and a prediction network. Based on a task input, the meta learner generates parameters for the 
feature layers in the prediction network so that the feature embedding can be accurately adjusted for that task. 
We show that TAFE-Net is highly effective in generalizing to new tasks or concepts and evaluate the TAFE-Net on 
a range of benchmarks in zero-shot and few-shot learning. Our model matches or exceeds the state-of-the-art on 
all tasks. In particular, our approach improves the prediction accuracy of unseen attribute-object pairs by 4 
to 15 points on the challenging visual attribute-object composition task.

<img src="figs/arch.jpg" width="750">


