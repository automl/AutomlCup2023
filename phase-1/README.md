# AutoML Cup Starter Kit

Welcome to the AutoML Cup Competition!

To get started:

1. Create an account on [Codabench](https://www.codabench.org/)
1. Register with the AutoML Cup Competition
1. Edit `submission/model.py`
1. Upload to Codabench under the `My Submissions` tab.

## Submission

The entrypoint `model.py` and `requirements.txt` can be added to a zip as follows:

```sh
cd submission/
zip ../submission.zip *
```

## Testing

To run the training and scoring programs locally (e.g., for the `splice` dataset):

```sh
export DATASET=splice
cd ingestion_program

python3 ingestion.py \
   --dataset_dir=../data/input_data/$DATASET \
   --output_dir=../output/ \
   --ingestion_program_dir=../ingestion_program_dir/ \
   --code_dir=../submission/ \
   --score_dir=../output/ \
   --temp_dir=../output/ \
   --no-predict
```

```sh
export DATASET=splice
cd scoring_program

python score.py \
   --prediction_dir=../output \
   --dataset_dir=../data/input_data/$DATASET \
   --output_dir=../output/score/
```
 ### Docker
There is also a Docker image provided that the true competition utilizes located in `docker/Dockerfile`. To have an equivalent environment, you may use this for your testing.

### Singularity
There is a singularity definition based on the docker image in `docker/singularity.def`. 
To compile it into an image execute the steps below. 
Make sure you have the correct requirements.txt in the directory of the singularity.dif file before executing the commands.

```sh
cd phase-1/docker/
sudo singularity build singularity.sif singularity.def
```
Code can be executed with the singularity image by calling 
```sh
singularity exec <path_to_singularity.sif>/singularity.sif <command>
```


## Datasets

Phase 1 comprises two 1D single-class classifcation problems, with the intention of developing methods that might only work (initially) for this problem type.
For both tasks, the data are formatted as a [datasets.Dataset](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) with the following fields:

-   `'input'` is a 1-dimensional numerical vector (either of constant or variable length)
-   `'label'` is the scalar output

### Setup Listops

Download [this](https://drive.google.com/file/d/1bgsd-nL8lKxGq9iw4vdZJ5hN_8XjMYOi/view?usp=sharing) and put `basic_train.tsv` and `basic_val.tsv` in `data/input_data/listops/listops/` before trying to run the ingestion program.

## Reference

You can refer to the source code at

-   Ingestion Program: `ingestion/ingestion.py`
-   Scoring Program: `scoring/score.py`
