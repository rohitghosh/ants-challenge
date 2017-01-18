# ants-challenge

The repo was created for submission of entry to the [Ants Challenge Part I](https://www.crowdai.org/challenges/ants-challenge-part-1)

In the first part of this challenge, we will focus on the task of identification and tracking of individual ants over time. The training data provides the coordinates of all the ants for a subset of the time frames, and the goal of the challenge is to predict the coordinates of all the ants for the rest of the time frames.

#### Output

The output was 2 vectors:
- V<sub>A</sub> : 72 element vector (71 ant-class + background), each element being a probability representing the probability of the class
- V<sub>B</sub> : 144 element vector representing X,Y coordinates of each class (along with background)

The ant, present in each patch was determined as classes having probability greater than 0.5 in V<sub>A</sub>. The corresponding coordinates were determined from V<sub>B</sub>.

#### Input

For training purposes, 2 methods were tried:

- Dividing images into patches
- Same as above, along with all the 71 bar-codes as 71 other channels

#### Architecture & Training

The architecture consists of simple Conv-BN-Maxpool based repeating blocks followed by fully connected layers. From the second last fully connected layer 2 layers branch out - one for class prediction and another for coordinate prediction , each of which are fully connected layer.

The class prediction output is trained using cross-entropy loss while the class boundary output is trained using regression loss. Loss from both the branches are summed up for back-propagation through the net.
 
