from typing import Iterable, Iterator
import json
import regex as re
from itertools import chain


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Args:
        - vocab: dict[int, bytes]
            A dictionary mapping token IDs to their corresponding byte sequences.
        - merges: list[tuple[bytes, bytes]]
            A list of merge operations that the tokenizer has learned.
        - special_tokens: list[str] | None = None
            A list of special tokens that the tokenizer should treat as single tokens.
        """
        # self.vocab = {k: v for k, v in vocab.items()}
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.merges = [
            {
                "ids": [self.vocab_inv[merge[0]], self.vocab_inv[merge[1]]],
                "values": b"".join(merge),
            }
            for merge in merges
        ]
        self.special_tokens = special_tokens
        if self.special_tokens is not None:
            # self.special_tokens = sorted(self.special_tokens, reversed=True)
            self.SPECIAL_TOKEN_PAT = f"({'|'.join(re.escape(token) for token in sorted(special_tokens, reverse=True))})"
            for special_token in special_tokens:
                special_token_byte = bytes(special_token, "utf-8")
                if special_token_byte not in self.vocab_inv:
                    idx = len(self.vocab)
                    self.vocab[idx] = special_token_byte
                    self.vocab_inv[special_token_byte] = idx
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def check_and_append_special_token(self, s, seq):
        if self.special_tokens:
            for special_token in self.special_tokens:
                if len(s) == len(special_token) and s in self.special_tokens:
                    seq.append(self.vocab_inv[bytes(special_token, "utf-8")])
                    return True
        return False

    def check_and_append_token_in_vocab(self, s, seq):
        if s in self.vocab_inv:
            seq.append(self.vocab_inv[s])
            return True
        return False

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the
        #same format that your BPE training code output) and (optionally) a list of special tokens.

        Args:
        - vocab_filepath: str
        - merges_filepath: str
        - special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
            vocab = {v: k for k, v in vocab.items()}

        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                a, b = line.strip().split(" ")
                merges.append([a, b])
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Given an input text, return a list of pre-tokenized words. This is a helper function for encode.
        """
        seq = []
        if self.special_tokens is None:
            text_chunks = [text]
        else:
            text_chunks = re.split(self.SPECIAL_TOKEN_PAT, text)
        for chunk in text_chunks:
            if self.check_and_append_special_token(chunk, seq):
                continue
            words = [bytes(word, "utf-8") for word in re.findall(self.PAT, chunk)]
            for word in words:
                if self.check_and_append_token_in_vocab(word, seq):
                    continue
                word_ids = self.bytes_to_tuples(word)
                for merge in self.merges:
                    if merge["values"] in word:
                        new_vocab_idx = self.vocab_inv[merge["values"]]
                        word_ids = self.replace_subsequence(
                            word_ids, merge["ids"], [new_vocab_idx]
                        )
                        if len(word_ids) < 2:
                            break
                seq.extend(word_ids)
        return seq

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required
        for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_strs = self.tuple_to_bytes(ids)
        return byte_strs.decode("utf-8", errors="replace")

    def str_to_byte(self, s: str):
        return bytes(s, "utf-8")

    def bytes_to_tuples(self, byte):
        return [self.vocab_inv[byte[i : i + 1]] for i in range(len(byte))]

    def tuple_to_bytes(self, t):
        return b"".join([self.vocab[i] for i in t])

    def is_subsequence(self, word, subseq):
        byte_subseq = self.tuple_to_bytes(subseq)
        return byte_subseq in word

    @staticmethod
    def replace_subsequence(lst, subsequence, replacement):
        """
            Replace all occurrences of subsequence in lst with replacement.

            Args:
            - lst: List of integers.
            - subsequence: List of integers representing the subsequence to replace.
            - replacement: List of integers representing the replacement sequence.

            Returns:
        - A list with the subsequence replaced.
        """
        i = 0
        while i <= len(lst) - len(subsequence):
            if lst[i : i + len(subsequence)] == subsequence:
                # Replace the subsequence with the replacement
                lst = lst[:i] + replacement + lst[i + len(subsequence) :]
                i += len(replacement)  # Move past the replacement
            else:
                i += 1
        return lst


if __name__ == "__main__":
    vocab_filepath = "/Users/byronzhang/Downloads/spring2024-assignment1-basics/tests/fixtures/train-bpe-reference-vocab.json"
    merges_filepath = "/Users/byronzhang/Downloads/spring2024-assignment1-basics/tests/fixtures/train-bpe-reference-merges.txt"
    tokenizer = Tokenizer.from_files(Tokenizer, vocab_filepath, merges_filepath)
    print(tokenizer.merges)
