# PII Insertion

Detailed explanations for the PII insertion pipeline.

## Step-by-step

Make sure to be located in the Git folder, at the root level.

1. Generate fake profiles using `fake_persona.py`: `python src/dataset/pii_insertion/fake_persona.py`.
2. Complete the notes using the script `pii_injection.py`. You can use `gemini-2.5-flash` or `Qwen3-32B-AWQ`. The corresponding job is in `src/jobs/pii_injection`. Gemini is much faster than local vLLM instances.
3. For manual insertion, run the file `manual_insertion.py`.
4. Perform random sampling to get the desired PII level and generate the dataset files for SFT with `sampling.py`: `python src/dataset/pii_insertion/sampling.py` or `sampling_manual.py` depending on the type of inserted PII you would like to use

## Validation

1. For fake profiles, use `persona_check.py` to check the distributions of each field. It will check and display if any PII is both in the training and the validation sets, as well as if values are duplicated inside the same dataset (outputs plots in `outputs/splits`)
2. For PII injection, use `dataset/plots/distrib_generated.py` to plot the distribution of PII types, then `evaluation_pii.py` to manually evaluate the notes.

