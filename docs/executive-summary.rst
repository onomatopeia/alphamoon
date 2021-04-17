Executive Summary
~~~~~~~~~~~~~~~~~

This report was commissioned to propose a method for classification of handwritten letters and digits in binary images based on a provided sample data set of 30134 examples of 56x56 images. Each image in the provided data set represents a single character, either a letter of the Latin alphabet or an Arabic digit. Images are labelled with 36 distinct labels.

Assumptions made during the initial development of a model were that each class represents a unique character. Under this assumption, one class, namely class no 30, contained only a single example calling for one-shot learning approaches to be employed. However, in the course of the study, it was determined that class 30 and class 14 both contain the images representing the same letter N. The two classes were merged for the further analyses.

The proposed model attempt to transform each example from an image space into a space of image embedding and classify those. The method comprises of two steps. In the first step, each image is translated into an embedding vector. This step is accomplished by means of a fully-connected feedforward neural network with a single hidden layer. In the second step, a k-NN classifier is trained to classify embeddings into classes.

Results of model performance analyses show that the proposed model attains 0.771 in class-weighted F1-score, 0.770 in precision and 0.775 in recall on a testing set constructed as a random third part of the provided data set.

The report also investigates the fact that the proposed model has limitations. The major limitation is that looking-alike characters are often confused by the model.

Recommendations discussed include:

* to revisit the training labels and provide a legitimate meaning for each class,
* to investigate a convolutional neural network for embedding learning,
* to examine alternative approaches that do not attempt to solve the few-shot learning problem but are excellent at character recognition instead.
