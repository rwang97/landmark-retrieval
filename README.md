# landmark-retrieval
In this paper, we extend two deep learning methods to perform image retrieval on Google Landmark Dataset v2. Image retrieval provides an efficient method that helps to search for related visual information within a large database. Image retrieval can be formatted as a representation learning problem where we construct an embedding space for querying similar landmark images. Our network uses ResNet-101 pre-trained on ImageNet for feature extraction and we apply two representation learning algorithms to maximize the intra-class compactness and the inter-class discrepancy of the embeddings extracted from the landmark images. We demonstrate that these two algorithms can be transferred to the image retrieval task beyond its application in the original domain, and one of the algorithms ArcFace obtains a superior retrieval performance.

<span>r</span><span>0.305</span>

Introduction
============

The primary goal of image retrieval is to query a base image by analyzing the relevance of all the contents in an image database and collecting data that is similar to the base image. A large-scale benchmark, the Google Landmarks Dataset v2 (GLDv2)  can be used to evaluate the performance and generalization of image retrieval techniques, and it contains more than 5 million images of human-made and natural landmarks worldwide.

In this paper, we implement two representation learning algorithms for landmark image retrieval. The first algorithm is inspired by *Generalized End-to-End Loss for Speaker Verification* , which was proposed to perform speaker verification by leveraging the centroids of the embedding vectors for different speakers to find representative clusters. The second algorithm, *Additive Angular Margin Loss (ArcFace)* , adds an angular margin to the angle between the features and target weights in each dimension of class, which modifies the cross entropy loss to achieve more distinguishable embeddings . To compare the two algorithms, we use the same ResNet-101  network pre-trained on ImageNet  as the encoder before the representation learning stage. We also design a variant of U-Net as the baseline to learn low-dimensional embeddings during image reconstruction.

Related Works
=============

Our work is closely related to image recognition, transfer learning, and representation learning. ResNet  has achieved great success when it was first proposed and has many variations such as ResNeXt  and Wide-ResNet . However, vanilla ResNet is still widely used to experiment with image-related tasks. Transfer learning  has been a successful technique that mitigates the issue of data scarcity and has demonstrated promising results in many deep learning tasks. Cross entropy loss is the most commonly used loss function in image recognition. However, it only focuses on the correctness of classification with a low level of feature discrimination. To increase the level of feature discrimination, many representation learning algorithms have been proposed. Center Loss  was introduced to learn and reduce the distance between the features and their corresponding class centers. Multiplicative angular margin and additive cosine margin were introduced in SphereFace  and CosFace  to further push the features in different classes into a more compact space. The L2-constrained Softmax Loss  introduced the restriction over the feature by normalization to improve the accuracy of face verification.

Methods and Algorithm
=====================

Our method is mainly composed of three components: Encoder, Projection Network, and Representation Learning Loss Function. Given an input batch of \(N \times M\) images, where \(N\) is the number of different landmark classes and \(M\) is the number of images per class, we can input the image \({\mathbf{x}}_{ji}\) into the encoder network to get an output vector \({\mathbf{r}}_{ji}\) of dimension 2048. We use a pre-trained ResNet-101 as our encoder network and replace its last fully connected layer with a sequence of *dropout*, *fully connected* and *batchnorm-1d* layers, which compose our projection network. The projection network maps \({\mathbf{r}}_{ji}\) to an embedding vector \({\mathbf{e}}_{ji}\) of dimension 512, and L2-normalization will be applied to \({\mathbf{e}}_{ji}\). At inference time, \({\mathbf{e}}_{ji}\) will be used for retrieving related images in the database. During training, we will further apply the loss function to learn distinctive embeddings that are useful for retrieval tasks.

Generalized End-to-End Loss (GE2E)
----------------------------------

Our first retrieval algorithm (loss function) is inspired by an existing approach to solve the speaker verification task . The main idea is to construct a similarity matrix \({\mathbf{S}}_{ji,k}\) which holds the cosine similarity between each image embedding \({\mathbf{e}}_{ji}\) and the centroid \({\mathbf{c}}_{k}\) of all the embeddings for each landmark class \(k\). The entries in the matrix will be scaled by two learnable parameters \(w\) and \(b\).

\[{\mathbf{S}}_{ji,k}=w\cdot \cos({\mathbf{e}}_{ji},{\mathbf{c}}_k) + b , \qquad \label{eqn:softmax}
    L({\mathbf{e}}_{ji})=-{\mathbf{S}}_{ji,j}+\log\sum_{k=1}^N\exp({\mathbf{S}}_{ji,k})
    \vspace{-2mm}\]

We use one of the two loss functions proposed in the paper – the “softmax” approach, where we try to make the embedding of each image close to the centroid of that landmark class’s embeddings but also far from other classes’ centroids. The whole process is shown in the figure below.

![Overall Process of GE2E ](images/g2e2framework.png "fig:") [fig:mesh1]

Additive Angular Margin Loss (ArcFace)
--------------------------------------

For the second retrieval algorithm, we use Additive angular Margin Loss (ArcFace), which is an approach originally leveraged to obtain distinguishable features in the face recognition task.

[ht] ![image](images/46.png)

[fig:arcfacelossframework]

Image features \({\mathbf{e}}_{ji}\) are firstly extracted by the encoder and projection network, and then we take the multiplication of normalized features and weights by using a fully connected layer to get the vector \(\cos(\theta)\) (logit), which measures the similarity of weights and features in each dimension of class. The key component of this algorithm is to add an Additive Angular Margin Penalty \(m\) to the similarity angle \(\theta\). The cosine value of the summation angle will be the new logit after re-scaling. The remaining procedures are the same as using the softmax loss.

\[{L} =-\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s(\cos(\theta_{y_i}+m))}}{e^{s(\cos(\theta_{y_i}+m))}+\sum_{j=1,j\neq  y_i}^{n}e^{s\cos\theta_{j}}} 
\label{eq:arcface}
\vspace{-1mm}\]

As for the geometric interpretation, since the features and weights are already normalized, the angle \(\theta\) can directly control the arc length and corresponding geodesic gap within neighbouring classes on the hypersphere.

Experiments and Discussion
==========================

Data and Training
-----------------

We perform our analysis of the two algorithms on GLDv2 , which is split into 3 sets of images: train, index, and test, each with 4132914, 761757 and 117577 images. Since GLDv2 was constructed in a noisy manner, we use a cleaned version of the train set , which contains 158047 landmark images with 81313 different classes. We adjust the size of all images by padding black and scaling them to \(224 \times 224\). Some data augmentations are applied during training, including random flip, brightness change and Gaussian noise. We fetch \(N \times M\) images as a batch into training, where \(N = 48\) different landmark classes and \(M = 5\) images per class. We chose 5 images per class since around \(30\%\) of landmark classes have fewer than or equal to 5 images, thus reducing the probability of having duplicates in a batch. Our encoder network ResNet-101 is pre-trained on ImageNet and the final embedding size is 512 for both representation algorithms. We train the networks using SGD with an initial learning rate of 0.01 and we decrease it by half after 25K steps.

We design a customized U-Net  as the baseline model, where the encoder is replaced by the same pre-trained ResNet-101 for a fair comparison. U-Net is a better variant of the traditional autoencoder , where the skip connections transfer some input features from encoder to decoder, adding more expressive power to the network. We train the U-Net with the reconstruction loss, trying to learn how to project images onto a latent space.

Mean Average Precision (mAP)
----------------------------

Mean average precision \(\mathrm{mAP}\) is the most common performance metric used in the context of object detection and information retrieval . We use \(\mathrm{mAP@100}\) to evaluate our models, where the precision is calculated using the top-100 ranked images. This metric is defined as:

\[\vspace{-10px}
\mathrm{mAP@100}  = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{\mathrm{min}(m_q, 100)} \sum_{k=1}^{\mathrm{min}(n_q, 100)} \mathrm{P}_q(k) \mathrm{rel}_q(k)
\vspace{5px}\]

Where \(Q\) represents the number of query images in the index set and \(\mathrm{mAP@100}\) represents the average precision(\(\mathrm{AP}\)). When calculating the \(\mathrm{AP}\) for each query, \(m_q\) refers to the total number of ground truth labels and \(n_q\) refers to the number of retrieved images. The precision at rank \(k\) is the number of correctly retrieved images divided by \(k\). The relevance function is an indicator function that outputs 1 if a retrieved image is relevant to the query and outputs 0 otherwise.

[ht!]

[tab:multireader]

We use \(\mathrm{mAP@100}\) as our metric to compare the performance. We first run our encoder network on full test and index set to extract image embeddings, and create a kNN (\(k=100\)) lookup for each test embedding by using the cosine distance between test and index embeddings. \(\mathrm{mAP}\) score is calculated based on the lookups. ArcFace outperformed the other two models to a large extent. Furthermore, we conduct analysis on the generalization and scalability of the three models. We find the performance degradation by calculating \(\mathrm{mAP@100}\) on different index set sizes and dividing them by \(\mathrm{mAP@100}\) at the smallest index set size. The generalization of ArcFace is superior to the other two as the score drops with the slowest rate against increasing index set size.

UMAP Embedding Projection Comparison
------------------------------------

The final 512-dimensional embeddings obtained by the three models have been projected to 2D using the UMAP algorithm . UMAP is a popular technique for dimension reduction and is better at preserving the global structure of the embeddings than t-SNE . We sample 10 landmark classes, each with around 50 images to extract the embeddings.

[fig:figure]

The baseline U-Net embeddings have a noisy embedding space where images from different classes can be projected to the same regions, even though we see some distinctive regions (green and yellow). The GE2E embeddings, on the other hand, demonstrate a larger intra-class compactness since the nature of GE2E is to push images close to the centroids of its class and further from the centroids of other classes. Therefore, the gap between the embeddings within the same class and their corresponding class centers is much smaller. ArcFace achieves a better inter-class discrepancy as it introduces the additive angular margin, which makes the geometric distance between the neighboring classes more evident. Different landmark classes are thus more distinguishable in the embedding space. Both GE2E and ArcFace show a higher level of feature discrimination, but overall ArcFace is more superior in terms of both intra-class compactness and inter-class discrepancy.

Summary
=======

We have extended two representation learning algorithms, GE2E and ArcFace, which were originally used for speech verification and face recognition respectively, to solve the task of landmark image retrieval. The final results demonstrate the capability of cross-domain-utilization of these two algorithms, indicating that they can be transferred to the image retrieval task and obtain satisfying performance. Based on extensive experiments, we have found that ArcFace outperforms the GE2E model in image retrieval and has better potential for other representation learning tasks.


