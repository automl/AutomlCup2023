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

To run the training and scoring programs locally (e.g., for the `pde` dataset):

```sh
export DATASET=pde
cd ingestion_program

python3 ingestion.py \
   --dataset_dir=../data/$DATASET \
   --datasets_root=../data/
   --output_dir=../output/ \
   --ingestion_program_dir=../ingestion_program_dir/ \
   --code_dir=../submission/ \
   --score_dir=../output/ \
   --temp_dir=../output/ \
   --no-predict
```

Note: TODO allow prediction via a validation set created from the training set. For now, you may create your own 'pseudo'-test set.

```sh
export DATASET=pde
cd scoring_program

python score.py \
   --prediction_dir=../output \
   --dataset_dir=../data/$DATASET \
   --output_dir=../output/score/
```

There is also a Docker image provided that the true competition utilizes located in `docker/Dockerfile`. To have an equivalent environment, you may use this for your testing.

## Datasets

-   CAMELYON17
-   Global WHEAT
-   PDE

### Metadata

We now supply the following properties via a metadata object during `Model.__init__()`.

```py
input_dimension: int
input_shape: InputShape
output_shape: OutputShape
output_type: OutputType
evaluation_metric: EvaluationMetric
training_limit_sec: int
```

### Setup

You can download the datasets here and place them in `phase-3/data/<dataset name>`:
https://drive.google.com/file/d/11UYW_rftUqFgWicc3-mUCzHxjf988glD/view?usp=sharing
https://drive.google.com/file/d/1Pec_sk4uyxIx-ToO0UhppIB2LUF47vyO/view?usp=sharing
https://drive.google.com/file/d/1k1S0_El8Ds09E9OQG2MVQN_PrVWkJPWI/view?usp=sharing

## Reference

You can refer to the source code at

-   Ingestion Program: `ingestion/ingestion.py`
-   Scoring Program: `scoring/score.py`
