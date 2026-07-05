from basics.tokenizer import tokenizer

VOCAB_PATH = 'data/tokenizer/TinyStories_diffusion_vocab.json'
MERGE_PATH = 'data/tokenizer/TinyStories_diffusion_merges.txt'
SPECIAL_TOKENS = ['<|endoftext|>', '<|mask|>']

def get_tokenizer (vocab_path=VOCAB_PATH, merge_path=MERGE_PATH, special_toekns=SPECIAL_TOKENS) -> tokenizer:
    return tokenizer.from_files(vocab_path, merge_path, special_toekns)