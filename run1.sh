#!/bin/bash
results=$"\n\n"
rm -f results/results.txt
rm -f phase-1/output/prediction
cp models/model1.py models/model.py
for dataset in listops; do
	rm -f phase-1/output/score/scores.json

	echo -e "\n=== $dataset ===\n"
	python3 phase-1/ingestion_program/ingestion.py \
		--dataset_dir=phase-1/data/input_data/$dataset \
		--output_dir=phase-1/output/ \
		--ingestion_program_dir=phase-1/ingestion_program_dir/ \
		--code_dir=models/ \
		--score_dir=phase-1/output/ \
		--temp_dir=phase-1/output/ \
		--time_budget=18000 &&
		python3 phase-1/scoring_program/score.py \
			--prediction_dir=phase-1/output \
			--dataset_dir=phase-1/data/input_data/$dataset \
			--output_dir=phase-1/output/score/ &&
		cat phase-1/output/score/scores.json &&
		results+=$dataset$" "$(<phase-1/output/score/scores.json)$"\n"
done
echo -e $results >results/results.txt
echo -e $results
rm -f phase-1/output/score/scores.json
rm -f models/model.py
