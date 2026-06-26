import re
import json
import time
import heapq
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

        pair_counts = {}
        pair_to_seqs = {}
        for idx, seq in enumerate(token_id_sequences):
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                if pair not in pair_to_seqs:
                    pair_to_seqs[pair] = set()
                pair_to_seqs[pair].add(idx)

        base_vocab_size = len(self.vocab)
        total_merges = vocab_size - base_vocab_size
        start_time = time.time()
        print(f"Starting BPE: {len(token_id_sequences)} sequences, {len(pair_counts)} unique pairs")

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        for new_id in range(base_vocab_size, vocab_size):
            best_pair = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                actual_count = pair_counts.get(pair, 0)
                if actual_count > 0 and actual_count == -neg_count:
                    best_pair = pair
                    break
            if best_pair is None:
                break

            step = new_id - base_vocab_size
            if step % 100 == 0 or new_id == vocab_size - 1:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed if elapsed > 0 else 0
                remaining = (total_merges - step - 1) / rate if rate > 0 else 0
                print(
                    f"Merge {step + 1}/{total_merges} | "
                    f"pair={best_pair} freq={pair_counts[best_pair]} | "
                    f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining"
                )

            self._replace_and_update_counts(
                token_id_sequences, best_pair, new_id,
                pair_counts, pair_to_seqs, heap
            )
            self.bpe_merges[best_pair] = new_id

        elapsed = time.time() - start_time
        print(f"Training complete: {len(self.bpe_merges)} merges in {elapsed:.1f}s")

        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    @staticmethod
    def _replace_and_update_counts(
        sequences, pair, new_id, pair_counts, pair_to_seqs, heap
    ):
        p0, p1 = pair
        affected_idxs = pair_to_seqs.pop(pair, set())
        pair_counts.pop(pair, None)
        changed_pairs = set()

        for idx in affected_idxs:
            seq = sequences[idx]
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == p0 and seq[i + 1] == p1:
                    if new_seq:
                        left = new_seq[-1]
                        old_left = (left, p0)
                        pair_counts[old_left] = pair_counts.get(old_left, 0) - 1
                        if pair_counts[old_left] <= 0:
                            pair_counts.pop(old_left, None)
                            pair_to_seqs.pop(old_left, None)
                        else:
                            changed_pairs.add(old_left)

                    if i + 2 < len(seq):
                        right = seq[i + 2]
                        old_right = (p1, right)
                        pair_counts[old_right] = pair_counts.get(old_right, 0) - 1
                        if pair_counts[old_right] <= 0:
                            pair_counts.pop(old_right, None)
                            pair_to_seqs.pop(old_right, None)
                        else:
                            changed_pairs.add(old_right)

                    new_seq.append(new_id)

                    if len(new_seq) >= 2:
                        new_left = (new_seq[-2], new_id)
                        pair_counts[new_left] = pair_counts.get(new_left, 0) + 1
                        if new_left not in pair_to_seqs:
                            pair_to_seqs[new_left] = set()
                        pair_to_seqs[new_left].add(idx)
                        changed_pairs.add(new_left)

                    if i + 2 < len(seq):
                        right = seq[i + 2]
                        new_right = (new_id, right)
                        pair_counts[new_right] = pair_counts.get(new_right, 0) + 1
                        if new_right not in pair_to_seqs:
                            pair_to_seqs[new_right] = set()
                        pair_to_seqs[new_right].add(idx)
                        changed_pairs.add(new_right)

                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            sequences[idx] = new_seq

        for p in changed_pairs:
            count = pair_counts.get(p, 0)
            if count > 0:
                heapq.heappush(heap, (-count, p))

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
