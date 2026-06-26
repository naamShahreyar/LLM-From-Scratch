import re
import json
from collections import Counter, deque

from tokenizer.base import Tokenizer


class BPETokenizer(Tokenizer):

    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}

    def train(self, text, vocab_size, allowed_special=["<|endoftext|>"]):
        tokens = self.pre_tokenize_text(text)

        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char
            for char in sorted(char for token in tokens for char in token)
            if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        token_id_sequences = [
            [self.inverse_vocab[char] for char in token] for token in tokens
        ]

        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_frequent_pairs(token_id_sequences, mode="most")
            if pair_id is None:
                break
            token_id_sequences = self.replace_pairs(
                token_id_sequences, pair_id, new_id
            )
            self.bpe_merges[pair_id] = new_id

        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def encode(self, text):
        tokens = self.pre_tokenize_text(text)

        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))
        return token_ids

    def decode(self, token_ids):
        out = []
        for tid in token_ids:
            if tid not in self.vocab:
                raise ValueError(f"Token ID {tid} not found in vocab.")
            tok = self.vocab[tid]

            if tok in ("\n", "\r"):
                out.append(tok)
            elif tok.startswith("Ġ"):
                out.append(" " + tok[1:])
            else:
                out.append(tok)
        return "".join(out)

    def tokenize_with_bpe(self, token):
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [
                char for char, tid in zip(token, token_ids) if tid is None
            ]
            raise ValueError(f"Token contains unknown characters: {missing_chars}")

        while len(token_ids) >= 2:
            pairs = set(zip(token_ids, token_ids[1:]))
            best_pair = min(
                (p for p in pairs if p in self.bpe_merges),
                key=lambda p: self.bpe_merges[p],
                default=None,
            )
            if best_pair is None:
                break

            new_id = self.bpe_merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(token_ids):
                if (
                    i < len(token_ids) - 1
                    and (token_ids[i], token_ids[i + 1]) == best_pair
                ):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            token_ids = new_tokens

        return token_ids

    @staticmethod
    def pre_tokenize_text(text):
        tokens = []
        parts = re.split(r"(\r\n|\r|\n)", text)

        for part in parts:
            if part == "":
                continue
            if part == "\r\n":
                tokens.append("\r")
                tokens.append("\n")
                continue
            if part == "\r":
                tokens.append("\r")
                continue
            if part == "\n":
                tokens.append("\n")
                continue

            pending_spaces = 0
            for m in re.finditer(r"( +)|(\S+)", part):
                if m.group(1) is not None:
                    pending_spaces += len(m.group(1))
                else:
                    word = m.group(2)
                    if pending_spaces > 0:
                        for _ in range(pending_spaces - 1):
                            tokens.append("Ġ")
                        tokens.append("Ġ" + word)
                        pending_spaces = 0
                    else:
                        tokens.append(word)
            for _ in range(pending_spaces):
                tokens.append("Ġ")
        return tokens

    @staticmethod
    def find_frequent_pairs(token_id_sequences, mode="most"):
        pairs = Counter(
            pair
            for token_ids in token_id_sequences
            for pair in zip(token_ids, token_ids[1:])
        )
        if not pairs:
            return None
        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'most' or 'least'.")

    @staticmethod
    def replace_pairs(token_id_sequences, pair_id, new_id):
        replaced_sequences = []

        for token_ids in token_id_sequences:
            dq = deque(token_ids)
            replaced = []

            while dq:
                current = dq.popleft()
                if dq and (current, dq[0]) == pair_id:
                    replaced.append(new_id)
                    dq.popleft()
                else:
                    replaced.append(current)

            replaced_sequences.append(replaced)
        return replaced_sequences

    def save(self, vocab_path, merges_path):
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(merges_path, "w", encoding="utf-8") as f:
            merges_list = [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.bpe_merges.items()
            ]
            json.dump(merges_list, f, ensure_ascii=False, indent=2)

    def load(self, vocab_path, merges_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            loaded_vocab = json.load(f)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            merges_list = json.load(f)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id
