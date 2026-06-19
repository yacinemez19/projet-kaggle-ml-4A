---
title: Projet Kaggle 2026
source: projet_kaggle_2026.pdf
---

*ModIA S8* — *2026*
MACHINE LEARNING — *Projet*

> *[Figure : logo INP ENSEEIHT]*

# Kaggle Project (binome)

Deep learning models are widely used for image classification. The goal of this project is to compare state-of-the-art models on the TILDA textile texture dataset. The link of the project is : https://www.kaggle.com/t/286b14f08b6941f3a1bb4b25e23f6f78

**Notice** : You should upload a report in **English**, together with your code on Moodle, for the validation of the course. Do not copy-paste other people's report or code.

**Deadlines** : Kaggle submission : 25 June 2026 | Report : 26 June 2026.

## 1 Learn how to use Pytorch

Refer to the website https://pytorch.org/tutorials/beginner/basics/intro.html.

## 2 Train state-of-the-art CNN models

The goal is to achieve a good classification accuracy on the test dataset, using Pytorch.

**Question 2.1 :** Pre-process your images so that they have the same input size to your model, e.g. use data augmentation.

**Question 2.2 :** Implement one or two models from the following list :

- LeNet Model [1]
- AlexNet Model [2]
- ResNet Model [3]
- or any choice available in the literature.

In your report, you should explain

- Why did you choose this particular network ?
- Give a precise definition of the model that you use, e.g. the number of layers, the type of each layer in the CNN, and a brief description on what is each layer doing.

**Question 2.3 :** Train your model using mini-batch SGD. Specify the optimization method which you use and report the total training time. To reduce the training time, you may use a GPU card.

**Question 2.4 :** Perform your parameter turning on a validation set to avoid over-fitting. Summarize your results in table/figure.

---

MACHINE LEARNING

# 3 Machine learning and AI safety

## 3.1 Bias introduction in TILDA

We try to understand how an image classification model (such as the ones trained in section 2) can be biased towards a specific outcome. To make things easier we consider the binary classification problem (2 classes : cauliflower and head cabbage).

**Question 3.1 :** Describe, with your own words, a real life situation in which there is, or there might be, bias. Comment on the consequences.

We propose to implement the following setup :

### 3.1.1 Biased Dataset Construction

We decide to concatenate to each of our images $x$ from TILDA a new set of variables $\epsilon$. As we shall specify, the second variable $\epsilon$ is a noise-like image, which is not directly computed from $x \in \mathbb{R}^{N \times N}$. Yet, it can introduce some bias if we choose the value of $\epsilon \in \mathbb{R}^{N \times N}$ to be strongly correlated to the label $y \in \{0, 1\}$ of each image $x$.

More precisely, we assume $(X, y) = ([x, \epsilon], y) \in \mathbb{R}^{2 \times N \times N} \times \{0, 1\}$ is a random sample of the modified dataset. We shall specify the value of $\epsilon$ using $y$. Let $p_0 \in [0, 1]$, $p_1 \in [0, 1]$ be two probabilities. Given $(x, y)$, the bias variable $S$ is defined as

$$S\{y = k\} \sim Bernoulli(p_k), \quad k \in \{0, 1\}.$$

This means that $\mathbb{P}(S = 1 | y = k) = p_k$ and $\mathbb{P}(S = 0 | y = k) = 1 - p_k$. Then we choose $\epsilon$ according to $S$ as below :

- $\epsilon = 0$ if $S = 0$
- $\epsilon \sim \mathcal{N}(0, I)$ if $S = 1$

where $I$ is the identity matrix of size $N \times N$.

In the extreme case where $p_0 = 0$ and $p_1 = 1$, one can ignore the original image $x$ and only use $\epsilon$ to predict $y$. In that case, our predictions are never using the image $x$ and we are only relying on the noise-like $\epsilon$ that we introduced to the dataset.

**Advice :** in Pytorch, we can build a biased dataset by simply adding the biased variables $\epsilon$ as a dedicated channel to each original image $x$ in the whole dataset.

**Question 3.2 :** Given $p_0$ and $p_1$, build a biased dataset from TILDA using the approach described in 3.1.1.

### 3.1.2 Model Bias Evaluation

We compare two different settings :

- $model_1$ is trained on an unbiased TILDA dataset ($p_0 = 1/2, p_1 = 1/2$)
- $model_2$ is trained on a biased version of TILDA ($p_0 = 0, p_1 = 1$)

---

MACHINE LEARNING

We finally compare the actual bias in both $model_1$ and $model_2$.

Assume $\hat{y}(X)$ is the prediction of a model computed from $X$. We shall separate the dataset set into 2 groups, one with $S = 0$, the other with $S = 1$. The bias of this model can be computed from a ratio of these two groups, using the $DI$ metric [4] (at the following link : https://github.com/wikistat/Fair-ML-4-Ethical-AI/tree/master/Propublica) defined as :

$$DI = \frac{\mathbb{P}(\hat{y}(X) = 1 | S = 0)}{\mathbb{P}(\hat{y}(X) = 1 | S = 1)}$$

In practice, a model is considered unbiased as long as the $DI$ metric is close to 1.

**Question 3.3 :** Study experimentally if $model_2$ is biased or not on test data. Compare you results to what you have with $model_1$.

**Question 3.4 :** Compare the accuracy scores on your test data for both $model_1$ and $model_2$ for the binary classification task.

**Question 3.5 :** Summarize your results into tables (e.g. $model_1$ and $model_2$, vs. DI and accuracy evaluated on training/validation sets). What can you conclude from the results ?

## 3.2 Bias Study in the Literature

**Question 3.6 (bonus) :** This article gives an overview of current AI system requirement in EU [4]. Read it carefully and write a short essay about one page through your critical thinking on "Est-ce qu'on peut faire confiance à mon modèle ?" from the perspective of model bias. You can also use other articles in the literature. More broadly, you may refer to the reports of CNIL [5].

# Références

[1] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11) :2278–2324, 1998.

[2] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Communications of the ACM*, 60(6) :84–90, 2017.

[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 770–778, 2016.

[4] https://hal.science/hal-03253111.

[5] https://www.cnil.fr/fr/intelligence-artificielle/guide/conformite-des-systemes-dia-les-autres-guides-outils-et-bonnes-pratiques.
