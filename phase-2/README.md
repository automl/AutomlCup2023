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

To run the training and scoring programs locally (e.g., for the `lego` dataset):

```sh
export DATASET=lego
cd ingestion_program

python3 ingestion.py \
   --dataset_dir=../data/$DATASET \
   --output_dir=../output/ \
   --ingestion_program_dir=../ingestion_program_dir/ \
   --code_dir=../submission/ \
   --score_dir=../output/ \
   --temp_dir=../output/ \
   --no-predict
```

You may create your own 'pseudo'-test set.

```sh
export DATASET=lego
cd scoring_program

python score.py \
   --prediction_dir=../output \
   --dataset_dir=../data/$DATASET \
   --output_dir=../output/score/
```

There is also a Docker image provided that the true competition utilizes located in `docker/Dockerfile`. To have an equivalent environment, you may use this for your testing.

## Datasets

Phase 2 comprises two 1D single-class classifcation problems (Splice, Lego) and a 2D regression problem (Protein), with the intention of developing methods that might only work (initially) for this problem type.
For both tasks, the data are formatted as a `Dict[str, ArrayLike]` with the following fields:

-   `'input'` is the input with the set input shape (see Metadata)
-   `'label'` is the output with the set output shape (see Metadata)

### Setup

You can download the datasets [here](https://drive.google.com/file/d/1pbAj9KdaBye6gOeEhfoj5fw2BY3PALC9/view?usp=sharing) and place them in `phase-2/data/`.


## Reference

You can refer to the source code at

-   Ingestion Program: `ingestion/ingestion.py`
-   Scoring Program: `scoring/score.py`
