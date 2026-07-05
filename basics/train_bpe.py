import regex as re
from collections import Counter
from tqdm import trange
import json

def train_bpe(input_path, vocab_size, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Initialize vocab as id -> bytes
    vocab = {x: bytes([x]) for x in range(256)}
    bias = len(vocab)
    for i, token in enumerate(special_tokens):
        vocab[bias + i] = token.encode("utf-8")

    merges = []
    loop_count = vocab_size - len(vocab)

    counter = Counter()
    with open(input_path, "r", encoding="utf-8") as f:
        buffer = []
        for line in f:
            if "<|endoftext|>" in line:
                # Split line in case there are multiple <|endoftext|> in one line
                parts = line.split("<|endoftext|>")
                for i, part in enumerate(parts):
                    if i == 0:
                        buffer.append(part)
                    else:
                        # Process accumulated document
                        text = ''.join(buffer)
                        for match in re.finditer(PAT, text):
                            token = match.group()
                            token_bytes = tuple(bytes([b]) for b in token.encode("utf-8", errors="ignore"))
                            counter[token_bytes] += 1
                        buffer = [part]  # Start new buffer with remainder after delimiter
            else:
                buffer.append(line)
        # Process any remaining text after the last <|endoftext|>
        if buffer:
            text = ''.join(buffer)
            for match in re.finditer(PAT, text):
                token = match.group()
                token_bytes = tuple(bytes([b]) for b in token.encode("utf-8", errors="ignore"))
                counter[token_bytes] += 1

    pair_counter = Counter()
    for token_bytes, freq in counter.items():
        for i in range(len(token_bytes) - 1):
            pair = (token_bytes[i], token_bytes[i + 1])
            pair_counter[pair] += freq

    # for _ in range(loop_count):
    for _ in trange(loop_count, desc="BPE merges"):
        if not pair_counter:
            break

        # 1) pick most frequent pair, tie‑break lexicographically
        max_count = pair_counter.most_common(1)[0][1]
        most_frequent_pair = max(
            pair for pair, count in pair_counter.items() if count == max_count
        )
        merge_pair = most_frequent_pair
        merges.append(merge_pair)

        # 2) add merged token to vocab
        new_token = b"".join(merge_pair)
        vocab[len(vocab)] = new_token

        new_counter = Counter()

        # 3) update counter and pair_counter incrementally
        for token_bytes, freq in counter.items():
            if merge_pair in zip(token_bytes, token_bytes[1:]):
                # 3a) remove OLD pairs
                for i in range(len(token_bytes) - 1):
                    p = (token_bytes[i], token_bytes[i + 1])
                    pair_counter[p] -= freq
                    if pair_counter[p] <= 0:
                        del pair_counter[p]

                # 3b) build merged token sequence
                merged = []
                i = 0
                while i < len(token_bytes) - 1:
                    if (token_bytes[i], token_bytes[i + 1]) == merge_pair:
                        merged.append(new_token)
                        i += 2
                    else:
                        merged.append(token_bytes[i])
                        i += 1
                if i == len(token_bytes) - 1:
                    merged.append(token_bytes[-1])

                merged = tuple(merged)
                new_counter[merged] += freq

                # 3c) add NEW pairs
                for i in range(len(merged) - 1):
                    p = (merged[i], merged[i + 1])
                    pair_counter[p] += freq
            else:
                new_counter[token_bytes] += freq

        counter = new_counter

    return vocab, merges

def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

def bytes_to_gpt2_unicode(byte_seq, byte_to_unicode):
    return ''.join(byte_to_unicode[b] for b in byte_seq)

def gpt2_unicode_to_bytes():
    byte_to_unicode = gpt2_bytes_to_unicode()
    return {v: k for k, v in byte_to_unicode.items()}

def unicode_str_to_bytes(token_str, unicode_to_byte):
    return bytes([unicode_to_byte[c] for c in token_str])

def save_vocab(vocab, save_path):
    # Prepare the mapping
    byte_to_unicode = gpt2_bytes_to_unicode()

    # Convert your vocab to the test format
    vocab_to_save = {}
    for token_id, token_bytes in vocab.items():
        token_str = bytes_to_gpt2_unicode(token_bytes, byte_to_unicode)
        vocab_to_save[token_str] = token_id

    # Save as JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)

def load_vocab(load_path):
    # Load vocab from file
    with open(load_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)

    unicode_to_byte = gpt2_unicode_to_bytes()
    vocab_loaded = {}
    for token_str, token_id in vocab_json.items():
        token_bytes = unicode_str_to_bytes(token_str, unicode_to_byte)
        vocab_loaded[token_id] = token_bytes

    return vocab_loaded

def save_merges(merges, save_path):
    byte_to_unicode = gpt2_bytes_to_unicode()
    with open(save_path, "w", encoding="utf-8") as f:
        for pair in merges:
            left = bytes_to_gpt2_unicode(pair[0], byte_to_unicode)
            right = bytes_to_gpt2_unicode(pair[1], byte_to_unicode)
            f.write(f"{left} {right}\n")

def load_merges(load_path):
    # Inverse mapping: unicode char → byte value
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    merges = []
    with open(load_path, encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                left, right = cleaned_line.split(" ")
                left_bytes = bytes([gpt2_byte_decoder[c] for c in left])
                right_bytes = bytes([gpt2_byte_decoder[c] for c in right])
                merges.append((left_bytes, right_bytes))
    return merges