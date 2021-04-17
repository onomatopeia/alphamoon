Executive Summary
~~~~~~~~~~~~~~~~~

This report was commissioned to propose a method for classification of handwritten characters in binary images based on a provided sample data set. Each images in the provided data set represents a single character, either a letter of the Latin alphabet or an Arabic digit. Images are labelled with 36 distinct labels.

Assumptions made during the initial development of a model were that each class represents a unique character. Under this assumption, one class, namely class no 30, contained only a single example calling for one-shot learning approaches to be employed.

Under the assumption that every two classes comprise unique and distinct elements

Once it was determined that class 30 and class 14 both contain the images representing the same letter N, the two classes were merged.



The proposed model

Results of data analyses show that a...
In particular, comparative performance is poor in the areas of...


The state of the dataset undermines the trust one would put in other people and faith into the quality of work they provide.

Whether there was a legitimate reason for splitting images representing the character "N" into two classes (14 and 30).

Recommendations discussed include:

* to revisit the classes and provide the meaning of each class. Understanding why the decision to split the same character into two classes was made could help developing a model that possesses such a discriminative power as well.

The report also investigates the fact that the proposed model has limitations. Some of the limitations include:

* looking-alike characters are often confused by the model - that resembles humans
* ... ?

