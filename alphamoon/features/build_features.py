from typing import Tuple

import torch.nn as nn
from torch import Tensor


class EmbeddingNet(nn.Module):
    """Network for learning embeddings. It contains two layers: a hidden layer
    with ReLU activation function and an output layer. Both layers are fully
    connected.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) \
            -> None:
        """Initializes the EmbeddingNet object.

        :param input_size: size (length) of input
        :param hidden_size: size of the hidden layer
        :param output_size: size of the output layer
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Implementation of the forward pass in this network module.

        :param x: data tensor
        :return: tensor resulting from the forward pass
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class TripletNet(nn.Module):
    """Triplet Neural Network module designed for training on triplets of
    (anchor, positive, negative) examples. The network aims to train an
    embedding neural network in such a way that embeddings of an anchor and
    of a positive example are close to each other, whereas the embedding
    of an anchor and a negative example are far away.
    """

    def __init__(self, embedding_net: EmbeddingNet) -> None:
        """Given an embedding network submodule, a triplet NN is initialized.

        :param embedding_net: embedding submodule
        """
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        """Implementation of the forward pass in this network module.

        :param anchor: anchor image
        :param positive: positive example image
        :param negative: negative example image
        :return: a triplet of tensors resulting from the forward pass
        """
        output1 = self.embedding_net(anchor.float())
        output2 = self.embedding_net(positive.float())
        output3 = self.embedding_net(negative.float())
        return output1, output2, output3
