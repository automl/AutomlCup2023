# Data

Phase 1 comprises two 1D single-class classifcation problems, with the intention of developing methods that might only work (initially) for this problem type.
For both tasks, the data are formatted as a [datasets.Dataset](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) with the following fields:

- `'input'` is a 1-dimensional numerical vector (either of constant or variable length)
- `'label'` is the scalar output
