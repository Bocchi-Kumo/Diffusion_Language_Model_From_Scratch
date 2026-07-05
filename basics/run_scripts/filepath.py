import os
from pathlib import Path
import sys
#===========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# project_root = Path(project_root) / 'cs336_basics'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
#===========================
# Vocab and merges path
VOCAB_DATAPATH = Path('data/tokenizer')
os.makedirs(VOCAB_DATAPATH, exist_ok=True)

TS_MERGES_PATH = VOCAB_DATAPATH / 'TinyStories_diffusion_merges.txt'
TS_VOCAB_PATH = VOCAB_DATAPATH / 'TinyStories_diffusion_vocab.json'

OWT_MERGES_PATH = VOCAB_DATAPATH / 'owt_train_merges.txt'
OWT_VOCAB_PATH = VOCAB_DATAPATH / 'owt_train_vocab.json'
#===========================
# Training and validation data path
TEXT_DATAPATH = Path('data/datasets')
os.makedirs(TEXT_DATAPATH, exist_ok=True)
IDS_PATH = TEXT_DATAPATH / 'token_ids'
os.makedirs(IDS_PATH, exist_ok=True)

TS_TRAIN_PATH = TEXT_DATAPATH / 'TinyStoriesV2-GPT4-train.txt'
TS_VALID_PATH = TEXT_DATAPATH / 'TinyStoriesV2-GPT4-valid.txt'

TS_TRAIN_IDS_PATH = IDS_PATH / 'TinyStoriesV2-diffusion-train_ids.bin'
TS_VALID_IDS_PATH = IDS_PATH / 'TinyStoriesV2-diffusion-valid_ids.bin'

OWT_TRAIN_PATH = TEXT_DATAPATH / 'owt_train.txt'
OWT_VALID_PATH = TEXT_DATAPATH / 'owt_valid.txt'
OWT_TRAIN_IDS_PATH = IDS_PATH / 'owt_train_ids.bin'
OWT_VALID_IDS_PATH = IDS_PATH / 'owt_valid_ids.bin'
#===========================
# Log path
LOGS_PATH = Path('data/logs')
os.makedirs(LOGS_PATH, exist_ok=True)
#===========================
# Model path
MODEL_PATH = Path('data/models')
os.makedirs(MODEL_PATH, exist_ok=True)
TS_MODEL_PATH = MODEL_PATH / 'tinysotry_diffusion_model.pt'
#===========================
SPECIAL_TOKENS = ["<|endoftext|>", "<|mask|>"]
#===========================
def assert_datasets_exist(required_files:list):
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"The following dataset files are missing.:\n"
            f"{chr(10).join(str(f) for f in missing_files)}"
        )