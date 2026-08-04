"""Shared configuration for the Subtask 1 (Arabic Dialect Sentiment Analysis) system."""
import os

SYSTEM_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(SYSTEM_DIR)
DATA_DIR = os.path.join(TASK_DIR, "Data")

TRAIN_PATH = os.path.join(DATA_DIR, "sent_train_v2.csv")
TEST_PATH = os.path.join(DATA_DIR, "predictions.csv")

CHECKPOINT_DIR = os.path.join(SYSTEM_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(SYSTEM_DIR, "outputs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Label space (fixed order used everywhere -- index <-> label mapping)
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# Dialect vocabulary for the optional dialect-embedding input feature (fgm_dialect
# technique). Lebanese has zero gold rows (only appears via self-training pseudo-labels)
# but still needs a vocabulary slot since it's present in test.
DIALECTS = ["Darija", "Egyptian", "Jordanian", "Saudi", "Lebanese"]
DIALECT2ID = {d: i for i, d in enumerate(DIALECTS)}

# Backbones chosen from the EDA + literature review:
#   - MARBERTv2: best reported performer (F1 79%) on the RANLP-2025 pilot of this exact
#     dataset (AHaSIS shared task), pretrained on a huge dialectal-Arabic Twitter corpus.
#   - CAMeLBERT-DA: pretrained on Dialectal Arabic, also the family used by the organizers
#     themselves as the official classifier for Subtask 2 -- proven strong on dialect
#     sentiment and gives ensemble diversity in tokenizer/pretraining corpus.
#   - AraBERTv2: MSA-centric pretraining gives a different inductive bias, useful ensemble
#     diversity, and the base register (hotel reviews) is not too far from MSA.
BACKBONES = {
    "marbertv2": "UBC-NLP/MARBERTv2",
    "camelbert_da": "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "arabertv2": "aubmindlab/bert-base-arabertv2",
}

MAX_LENGTH = 128          # EDA: median ~16-18 words, max ~123 words -> 128 subword tokens is ample
NUM_FOLDS = 5
SEED = 42
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

GPU_DEVICE = "cuda:0"  # NOTE: run with CUDA_VISIBLE_DEVICES=1 so this maps to the free physical GPU 1
