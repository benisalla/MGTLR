import unicodedata
import regex as re
import sys
import time

class MGTokenizer:
    def __init__(self, pattern=None):
        self.merges = {}
        self.pattern = (
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
            if pattern is None
            else pattern
        )
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self.get_vocab()

    def get_stats(self, ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def replace_control_characters(self, s: str) -> str:
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch)
            else:
                chars.append(f"\\u{ord(ch):04x}")
        return "".join(chars)

    def render_token(self, t: bytes) -> str:
        s = t.decode("utf-8", errors="replace")
        s = self.replace_control_characters(s)
        return s

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                self.get_stats(chunk_ids, stats)

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                current_time = time.strftime('%H:%M:%S')
                progress = (i + 1) / num_merges * 100
                print(
                    f"{current_time} - \033[95mProgress:\033[0m \033[96m{progress:.2f}%\033[0m | "
                    f"\033[94mMerge {i+1}/{num_merges}:\033[0m [{pair[0]} + {pair[1]}] -> [{idx}] | "
                    f"(\033[92m{self.render_token(vocab[idx])}\033[0m) with \033[93m{stats[pair]}\033[0m occurrences",
                    flush=True
                )

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _process_bytes(self, _bytes):
        ids = list(_bytes)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids

    def _process_text(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._process_bytes(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text):
        special = self.special_tokens
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self._process_text(part))
        return ids

    def get_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = self.render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = self.render_token(self.vocab[idx0])
                    s1 = self.render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self.get_vocab()
