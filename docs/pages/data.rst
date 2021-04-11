Image size and data augmentation
--------------------------------

I chose to first scale images so that the shorter edge is equal to the input size that a network accepts, and then crop them square. The input size of my network is 224 x 224, as for the majority of famous networks. This has a practical dimensions, since I can safely assume datasets are created with this size in mind.

For the training set I added:

- a random vertical flip, as we want a network to learn to recognise dog breeds from both sides and it is safe to assume that animals are vertically approximately symmetric,
- a random rotation (from -45 to 45 deg) to make the network more resilient towards varying orientation of a dog with respect to a camera.

I didn't add any transformations to validation and test sets as they should be as close as possible to the actual data.