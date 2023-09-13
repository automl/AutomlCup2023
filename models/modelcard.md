### Leaderboard names

##### Phase 1 leaderboard username: alexander

##### Phase 2 leaderboard username: alexander

##### Phase 3 leaderboard username: alexander

ML Lab Freiburg - AutoML Cup Team, arjun, sukthank, or Johannes Hog if alexander isn't on the leaderboard

### Description:

We created a taxonomy that determines the task type based on the input shape and trains multiple models that work well
for this task. Afterwards, we use greedy ensemble search to find a good ensemble of the trained models. We consider two
versions of each model, the one after the epoch with the lowest validation loss and the one after the final epoch in the
ensemble. For each model we run a portfolio of diverse hyperparameter configurations to cover the needs of different
tasks. The list of models we use consists of MultiRes, UNet, EfficientNet and others thus covering a wide variety of
potential tasks and architecture types. The pretrained [UNet]( https://github.com/milesial/Pytorch-UNet) is a widely
used python implementation of a unet that was
trained on a car segmentation task, we warmstart our training with the weights of that segmentation task and replace the
first/last layer of the UNet if necessary. This allows us to generalize and be applicable to any other task that has a
2D output where the width and height of the output are the same as the input. The pretrained EfficientNet and Vision
Transformer and their weights are part of torchvision. We again warmstart our training with these weights and replace
the last layer (and the first if necessary). This approach generalizes to any image classification/regression task. We
resize the images to 224 pixels for the Vision Transformer. With the use of pretrained weights we effectively use
open-source pretrained model weights to develop a robust and widely applicable ensemble selection scheme.
