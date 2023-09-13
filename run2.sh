#!/bin/bash
results=$"\n\n"
rm -f results/results.txt
rm -f phase-2/output/prediction.npz
cp models/model2.py models/model.py
for dataset in splice protein lego; do
	rm -f phase-2/output/score/scores.json

	echo -e "\n=== $dataset ===\n"
	python3 phase-2/ingestion_program/ingestion.py \
		--dataset_dir=phase-2/data/$dataset \
		--output_dir=phase-2/output/ \
		--ingestion_program_dir=phase-2/ingestion_program_dir/ \
		--code_dir=models/ \
		--score_dir=phase-2/output/ \
		--temp_dir=phase-2/output/ \
		--time_budget=36000 &&
		python3 phase-2/scoring_program/score.py \
			--prediction_dir=phase-2/output \
			--dataset_dir=phase-2/data/$dataset \
			--output_dir=phase-2/output/score/ &&
		cat phase-2/output/score/scores.json &&
		results+=$dataset$" "$(<phase-2/output/score/scores.json)$"\n"
done
echo -e $results >results/results.txt
echo -e $results
rm -f phase-2/output/score/scores.json
rm -f models/model.py
