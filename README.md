# CLGSI with LCTW

Source code for **Low-confidence task-aligned reweighting improves zero-boundary decisions in CLGSI-based multimodal sentiment analysis**.

LCTW adds a bounded, MAE-aligned gradient at the fused sentiment prediction near the zero boundary. It leaves the forward loss value and inference architecture unchanged. This repository contains the model, LCTW implementation, dataset loader, training and evaluation code. Datasets, pretrained models, experiment logs and manuscript files are not distributed here.

## Environment

The experiments used Linux, Python 3.8.10, PyTorch 2.4.1 with CUDA 12.1, Transformers 4.36.2, NumPy 1.24.4 and scikit-learn 1.3.2. Use a separate environment:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Training uses float32 without automatic mixed precision. The default device is `cuda:0`; `--device cpu` is available for small implementation checks. CPU checks do not reproduce the GPU experiments.

## Data and pretrained text encoder

Obtain the processed, unaligned MOSI or MOSEI features through the [MMSA data instructions](https://github.com/thuiar/MMSA), subject to the dataset provider's terms. The dataset pickle must contain `train`, `valid`, and `test` splits. Each split needs `text_bert`, `audio`, `vision`, `audio_lengths`, `vision_lengths`, and `regression_labels`.

`text_bert` has shape `[N, 3, L]` and stores token IDs, attention masks, and token-type IDs. Audio/visual feature dimensions are 5/20 for MOSI and 74/35 for MOSEI. The processed features and split membership must match the experimental setup; substituting a different feature release changes the experiment.

Download [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) to a local directory containing its configuration, tokenizer and weights. Pass that directory with `--bert`. The code loads it locally and does not distribute those weights.

## Training and evaluation

Run the baseline and LCTW with the same dataset, text-encoder files and seeds:

```bash
python train.py --dataset mosi --data datasets/mosi.pkl --bert checkpoints/bert-base-uncased --method clgsi
python train.py --dataset mosi --data datasets/mosi.pkl --bert checkpoints/bert-base-uncased --method lctw
python train.py --dataset mosei --data datasets/mosei.pkl --bert checkpoints/bert-base-uncased --method clgsi
python train.py --dataset mosei --data datasets/mosei.pkl --bert checkpoints/bert-base-uncased --method lctw
```

The defaults use seeds `10111 10112 10113`, at most 40 epochs, and early stopping after eight epochs without improvement. Checkpoint selection minimizes validation MAE rounded to four decimal places and retains the earliest checkpoint on ties. Test metrics are evaluated after selection. The training batch sizes are 64 for MOSI and 128 for MOSEI. LCTW uses `tau=0.25` and `rho=0.10`; the contrastive-loss coefficients are 0.95 and 0.48 respectively. Other hyperparameters are in `config/config_regression.py`.

Each run saves its configuration, selected checkpoint and validation/test metrics under `outputs/<dataset>/<method>/<seed>/`. A separate file reports the arithmetic mean of test metrics across the requested seeds. Existing run directories are not overwritten; use a new `--output` directory to repeat a run.

Acc-2, weighted F1 and Acc-7 are stored as fractions; multiply by 100 to express percentages. Has0 includes zero targets and uses `>= 0` for positive predictions and labels. Non0 excludes zero targets and uses `> 0`. MAE and correlation are reported without percentage scaling. Acc-7 clips predictions and targets to `[-3, 3]` and rounds them to integer classes.

## Implementation

- `lctw.py`: zero-forward surrogate and contrastive-loss adapter.
- `models/`: CLGSI network and contrastive loss.
- `config/`: dataset-specific model and optimizer settings.
- `data/load_data.py`: processed feature loader.
- `train.py`: training, validation-based selection and final evaluation.
- `metrics.py`: unrounded evaluation metrics.

The training entry point consolidates the experimental training procedure into a standalone script. Randomness, hardware, libraries and input features can affect the resulting metrics; running the script does not guarantee identical rounded scores.

## Acknowledgments

The backbone is based on [CLGSI](https://github.com/AZYoung233/CLGSI), commit `c1070c8267f255a772fa889493eab2d9e3c19c91`, introduced in [CLGSI: Contrastive Learning Guided by Sentiment Intensity for Multimodal Sentiment Analysis](https://aclanthology.org/2024.findings-naacl.135/). Its code also acknowledges [Self-MM](https://github.com/thuiar/Self-MM). The upstream MIT copyright and permission notice are retained in `LICENSE`. LCTW is the extension provided by this work; CLGSI is not a newly proposed backbone here.
