# AutoML Cup Submission - ML Lab Freiburg - AutoML Cup Team

Authors: Johannes Hog, Alexander Pfefferle

## The competition

The [AutoML Cup](https://2023.automl.cc/competitions/automl-cup/) was part of the AutoML Conference 2023.
It was split up into 3 phases, each expanding the scope of the previous phase:

+ [Phase 1](https://www.codabench.org/competitions/984/): 1D tasks
+ [Phase 2](https://www.codabench.org/competitions/1158/): any dimension
+ [Phase 3](https://www.codabench.org/competitions/1232/): any dimension, reduced labels

## Description of Solution

We created a taxonomy that determines the task type based on the input shape and trains multiple models that work well
for this task. Afterwards, we use greedy ensemble search to find a good ensemble of the trained models. We consider two
versions of each model, the one after the epoch with the lowest validation loss and the one after the final epoch in the
ensemble. For each model we run a portfolio of diverse hyperparameter configurations to cover the needs of different
tasks. The list of models we use consists of MultiRes, UNet, EfficientNet and others thus covering a wide variety of
potential tasks and architecture types. The pretrained [UNet](https://github.com/milesial/Pytorch-UNet) is a widely
used Python implementation of a UNet that was trained on a car segmentation task.
We warmstart our training with the weights of that segmentation task and replace the
first/last layer of the UNet if necessary. This allows us to generalize and be applicable to any other task that has a
2D output where the width and height of the output are the same as the input. The pretrained EfficientNet and Vision
Transformer and their weights are part of torchvision. We again warmstart our training with these weights and replace
the last layer (and the first if necessary). This approach generalizes to any image classification/regression task. We
resize the images to 224 pixels for the Vision Transformer. With the use of pretrained weights we effectively use
open-source pretrained model weights to develop a robust and widely applicable ensemble selection scheme.

## Datasets

We have copied the starter kit code for each of the three phases of the competition and put them
into `phase-1/`, `phase-2/` and `phase-3/` respectively. Each of these folders have a `README.md` file containing
instructions on how to download the datasets of that phase. Please follow these instructions. Additionally, you need
to put the datasets in the `data/datasets/competition/<dataset_name>/` folders. For phase 2 you have to additionally
execute
```sh
python data/create_phase-2_splits.py --dataset <dataset name>
```
to split the datasets into training and validation sets.
If you follow these instructions you would have two copies of all the datasets. You might want to avoid this by saving
the datasets first into the `data/datasets/competition/<dataset_name>/` folders and then creating symlinks to the data
folders in each phase.

## Running the code

### Requirements

You need a Python installation of version 3.10

```sh
python -m pip install -r models/requirements.txt
```

### Singularity
To create a singularity container from the definition file execute:
```sh
sudo singularity build singularity.sif singularity.def
```
When executing a Python command with singularity add the following to the Python command
```sh
singularity exec --nv -H <path_to_repository> singularity.sif python file.py
```
### Run single model

To run a single model on a given dataset you can execute:

```sh
python run_config.py --model <model name> --suite <phase-\d> --dataset_id <dataset name> --time_budget <budget>
```

You can find the names of the models in the model files. The model files are in `models/basemodels/`.

### HPO

We used HPO to find promising candidates for our portfolios. We tried to select dissimilar configurations from these
candidates to improve generalization.
You can execute the HPO code with:

```sh
python run_hpo.py --model <MODEL> --suite <phase-\d> --dataset_id <dataset id> --time_budget <total run time> --n_trials <number of trials>
```

There are also optional arguments which allow you to use cross-validation instead of the holdout default by specifying a
number of folds via the `--cv` parameter.
Additionally, it is also possible to use multi-fidelity on the number of datapoints with the `--multifidelity` flag.
For small datasets like splice we make use of 5-fold cross-validation while we applied multi-fidelity to slow datasets
like lego.

### Run submission

`./run1.sh` will copy `models/model1.py` to `models/model.py` and execute the ingestion and submission code for all
datasets in phase 1 using the code in `models/` as a submission.
`./run2.sh` and `./run3.sh` will do the same for phase 2 and phase 3 respectively.

## Submissions

Use `./zip.sh` to create the submissions for codabench.
It will create `phase1.zip` and `phase2.zip`, since our approach for phase 2 and phase 3 uses the exact same code you
can use `phase2.zip` as a submission for both phase 2 and phase 3.

The final submissions on codabench did not use the newest version of our code found in `models/`. The code of these
submission is zipped and in the `official_submissions/` folder.
