Method
======

The method comprises of two steps.

* In the first step, each image is translated into an embedding.
* In the second step, a k-NN classifier is trained to classify embeddings.

Image Embedding Learning
------------------------
Embedding Network
~~~~~~~~~~~~~~~~~

Image embedding is retrieved as the output of a small neural network constructed as a fully connected feedforward neural network with one hidden layer and ReLU activation function.

.. figure:: images/EmbeddingNet.png
   :alt: Fully connected feedforward neural network for embedding learning
   :width: 50%
   :align: center

   Fully connected feedforward neural network for embedding learning

Triplet Network
~~~~~~~~~~~~~~~

In order to learn the embeddings, a Triplet Network is used. A Triplet Network accepts as an input a triplet of examples, where:

* the first element is an anchor,
* the second element is a positive example i.e. another image from the same class,
* the third element is a negative example i.e. an image from a different class.

The goal is to learn the embeddings so that the L\ :sub:`2`\  distances between classes is maximized, that is the distance between an anchor and a positive example is small whereas the distance between an anchor and a negative example is large.

.. figure:: images/TripletNet.png
   :alt: Triplet-loss Network for Learning Image Embeddings
   :width: 50%
   :align: center

   Triplet-loss Network for Learning Image Embeddings

.. figure:: images/sample_triplet.png
   :alt: Sample triplet consisting of an anchor image, a positive example, and a negative example
   :width: 50%
   :align: center

   Example triplet consisting of an anchor image, a positive example, and a negative example

A loss function utilized for training the Triplet Network is Triplet Loss, defined as:

.. math::

   L = \max( d(a, p) - d(a,n) + margin, 0)

where

* :math:`d` is the distance function
* :math:`a` is an anchor
* :math:`p` is a positive example
* :math:`n` is a negative example,
* :math:`margin` is a hyperparameter that defines how far away the classes should be.

Consult the :mod:`alphamoon.features.build_features` module for an implementation of the above networks.

k-Nearest Neightbors Classifier
-------------------------------

A k-nearest neighbors algorithm was chosen as a method for deciding to which class a given example belongs to.

The input consists of the k closest training examples in data set. In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

An :class:`sklearn.neighbors.KNeighborsClassifier` implementation of k-NN algorithm was utilized.