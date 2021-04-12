Transfer learning
-----------------

Our dataset is small (20K images), but the dataset is similar to ImageNet on which Inception (and all other Torch models) was pretrained. Thus we should change the end of the ConvNet:

- freeze all the weights from the pretrained network,
- slice off the end of the neural network,
- add a new fully connected layer that matches the number of classes in the new data set,
- randomize the weights of the new fully connected layer,
- train the network to update the weights of the new fully connected layer.

We don't need to explicitly randomize the weights of the new fc layers, because the weights are randomized by default when a new Linear layer is created.

We need to create two fully connected layers

- one for a final classifier (this is common accross all networks)
- and one for the auxiliary classifier that Inception v3 model uses to improve its training.

Freezing the weights in the pretrained network prevents overfitting, which otherwise would be possible since our dataset is small.

The choice of Inception network is motivated by the fact that it outperformed other networks (AlexNet, VGG, etc) in ImageNet Large Scale Visual Recognition Competition 2014 in the task of classification. Moreover, as mentioned at the beginning of this notebook, ImageNet contains 118 dog breed classes, so these pretrained networks actually already know how to recognise dogs. Now it is only the matter of reassining which features correspond to each class, which is done in a final linear classificiation layer. Thus the above approach seems suitable to the problem at hand.


Comparison of various networks: https://arxiv.org/pdf/1605.07678.pdf

Siamese netowork
----------------

Since there is a limited number of examples for each class, we could resort to one(few)-shot learning where Siamese networks are a primary choice nowadays.

TODO: transfer learning and siamese networks?

Reasons to Use Siamese Neural Network (from `internet <https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example>`_):

- Needs less training Examples to classify images because of One-Shot Learning
- Learn by Embedding of the image so that it can learn Semantic Similarity
- It helps in ensemble to give the best classifiers because of its correlation properties.
- Mainly used for originality verification .

Potential augementations:
- translation
- very subtle rotation (up to say 5 degrees)
- aspect-ratio preserving scaling
- combinations of the above

