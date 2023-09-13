pip3 install -r $1/requirements.txt && \
 python3 /app/program/ingestion.py \
    --dataset_dir=$2 \
    --output_dir=/app/output/ \
    --ingestion_program_dir=/app/program/ \
    --code_dir=/app/ingested_program/ \
    --score_dir=/app/output/ \
    --temp_dir=/app/output/
