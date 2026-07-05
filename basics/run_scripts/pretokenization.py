import os
from typing import BinaryIO
import numpy as np
#===========================================
from filepath import *
from basics.run_scripts.filepath import *
from basics.tokenizer import tokenizer
#===========================================
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenization(input_path, output_path, tokenizer: tokenizer, num_processes = 1):
    '''
    input_path: A .txt file path.
    output_path: A .bin file path.
    num_processes: How many chunks would be splited to save the usage of memory.
    '''
    with open(input_path, "rb") as f, open(output_path, "wb") as out_f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            token_ids = tokenizer.encode(chunk)
            arr = np.array(token_ids, dtype=np.uint16)
            arr.tofile(out_f)  # Write directly to output file in binary format
    return True

# TinyStories
assert_datasets_exist([TS_VOCAB_PATH, TS_MERGES_PATH, TS_TRAIN_PATH, TS_VALID_PATH])
TS_tokenizer = tokenizer.from_files(vocab_filepath=TS_VOCAB_PATH, merges_filepath=TS_MERGES_PATH, special_tokens=SPECIAL_TOKENS)
pre_tokenization(input_path=TS_TRAIN_PATH, output_path=TS_TRAIN_IDS_PATH, tokenizer=TS_tokenizer, num_processes=3)
pre_tokenization(input_path=TS_VALID_PATH, output_path=TS_VALID_IDS_PATH, tokenizer=TS_tokenizer, num_processes=1)
print('Tiny Stories get tokenized into IDS')

# # OWT
# OWT_tokenizer = tokenizer.from_files(vocab_filepath=OWT_VOCAB_PATH, merges_filepath=OWT_MERGES_PATH, special_tokens=SPECIAL_TOKENS)
# pre_tokenization(input_path=OWT_TRAIN_PATH, output_path=OWT_TRAIN_IDS_PATH, tokenizer=OWT_tokenizer, num_processes=15)
# pre_tokenization(input_path=OWT_VALID_PATH, output_path=OWT_VALID_IDS_PATH, tokenizer=OWT_tokenizer, num_processes=1)
# print('OWT get tokenized into IDS')