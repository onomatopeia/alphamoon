Training
========

Training of a model was executed in two phases. In the first phase, the objective was to determine the best hyperparameters for the model. For that purpose the provided dataset was split into training and testing subsets; the testing subset served the purpose of assessing the generalization ability of a trained model. In the second phase, the objective was to train a final model using the predetermined hyperparameters' values.

1. Read the data
2. Reassign class 30 to 14
3. Split the data into the training and testing sets

   The provided data set was divided into a training and a testing sets in the proportion 2:1. To ensure proportional representation of all classes in both sets, a :class:`sklearn.model_selection.StratifiedShuffleSplit` was utilized to accomplish the split.

4. Create an instance of the Triplet Network
5. Define:

   a. number of epochs
   b. margin for the triplet loss
   c. loss function (:class:`torch.nn.TripletMarginWithDistanceLoss`)
   d. optimizer (:class:`torch.optim.Adam`)

6. Move the model to CUDA if CUDA device is available.
7. [Optional] Apply data augmentation transformations to the images in the training set

   Considered data augmentation included applying a random selection of transformations to an image:

   * random rotation of up to 10 degrees,
   * random horizontal and vertical scaling by the factor between 0.88 and 1.12, independent of each other,
   * random horizontal and vertical translation by up to 7 pixels in both directions, independent of each other.

   Each transformation was applied independently with probability 0.5.

   Consult the :meth:`alphamoon.data.make_dataset.get_transformation_matrix` function for the implementation.

8. Split the training data into training and validation sets

   The training dataset was further split into training and validation sets in the proportion 4:1. The role of the validation set was to ensure early stopping in case the Triplet Network started to overfit during its training.

   Consult the :class:`alphamoon.data.make_dataset.TripletDataset` class for triplets generation and :meth:`alphamoon.data.make_dataset.get_data_loaders()` for train-validation split implementation.

9. Create data loaders for both training and validation sets

   Consult the :mod:`alphamoon.data.make_dataset` module for the implementation data loaders cration and the triplet dataset :class:`alphamoon.data.make_dataset.TripletDataset`.

10. Learn image embeddings by training the Triplet Model
11. Train the k-NN classifier for various k and choose k for which F1 is maximized

   Odd numbers between 1 and 15 were used for evaluation. Even numbers were not considered, since by rule they yielded worse results than surrounding odd numbers (the majority vote is undetermined if votes split 50:50 among an even number of voters). Numbers higher than 15 were not considered as they yielded numerical error indicating that such higher numbers might generate unstable or undetermined results.

12. Save the resulting model to a file

Point 3 was skipped in the training of the final model.

For the implementation of phase 1 see :meth:`alphamoon.models.train_model.determine_best_model`, whereas phase 2 implementation can be found in :meth:`alphamoon.models.train_model.train_final_model`.

