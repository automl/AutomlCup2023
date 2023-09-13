#!/bin/bash
cd models
rm -rf base_models/models
rm -rf base_models/__pycache__/
rm -rf base_models/neural_nets/__pycache__/
rm -rf base_models/utils/
cp model1.py model.py
zip -r ../phase1.zip model.py requirements.txt modelcard.md preprocessing.py base_models configs p1metadata.py -x "base_models/trained_models/*"
cp model2.py model.py
zip -r ../phase2.zip model.py requirements.txt modelcard.md preprocessing.py base_models configs -x "base_models/trained_models/*"
rm -f models/model.py
cd ..
