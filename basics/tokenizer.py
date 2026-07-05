from basics import train_bpe

import regex as re
from typing import List, Iterable


# GPT‑2 / tiktoken pre-tokenization regex
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None
        """
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens: List[str] = special_tokens or []
        self._special_tokens_set = set(self.special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = train_bpe.load_vocab(vocab_filepath)
        merges = train_bpe.load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    # -------- internal helpers --------

    @staticmethod
    def _split_with_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
        if not special_tokens:
            return [text]
        specials_sorted = sorted(special_tokens, key=len, reverse=True)
        segments: List[str] = []
        i = 0
        buf: List[str] = []
        while i < len(text):
            matched = False
            for st in specials_sorted:
                if text.startswith(st, i):
                    if buf:
                        segments.append("".join(buf))
                        buf = []
                    segments.append(st)
                    i += len(st)
                    matched = True
                    break
            if not matched:
                buf.append(text[i])
                i += 1
        if buf:
            segments.append("".join(buf))
        return segments

    @staticmethod
    def _get_pairs(tokens: List[bytes]):
        # generator to avoid building a list each time
        for i in range(len(tokens) - 1):
            yield tokens[i], tokens[i + 1]
 
    def _bpe_merge(self, tokens: List[bytes]) -> List[bytes]:
        # in-place style loop, no nested function creation
        merges_dict = self.merges_dict
        while True:
            if len(tokens) < 2:
                break
            min_rank = float("inf")
            min_pair = None
            for pair in self._get_pairs(tokens):
                rank = merges_dict.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None:
                break
            new_tokens: List[bytes] = []
            i = 0
            last = len(tokens) - 1
            while i < len(tokens):
                if i < last and (tokens[i], tokens[i + 1]) == min_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    # -------- public API --------

    def encode(self, text: str) -> List[int]:
        segments = self._split_with_special_tokens(text, self.special_tokens)

        byte_tokens: List[bytes] = []
        for seg in segments:
            if seg in self._special_tokens_set:
                # whole special token as one bytes token
                byte_tokens.append(seg.encode("utf-8"))
            else:
                # apply GPT‑2 regex to this non‑special segment
                for m in GPT2_PAT.findall(seg):
                    seg_bytes = m.encode("utf-8")
                    sub_tokens = [bytes((b,)) for b in seg_bytes]
                    sub_tokens = self._bpe_merge(sub_tokens)
                    byte_tokens.extend(sub_tokens)

        ids: List[int] = [self.reverse_vocab[token] for token in byte_tokens]
        return ids

    def encode_iterable(self, iterable: Iterable[str]):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")