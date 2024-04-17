from regex import findall
from collections import Counter, defaultdict
import os

def encode(s: str):
    return s.encode("utf-8")

def bytes_to_tuples(b, vocab, special_tokens):
    t = tuple(b)
    num_special_tokens = len(special_tokens)
    for i in range(num_special_tokens):
        special_token_encoded = encode(vocab[i+256])
        if special_token_encoded in b:
            t = replace_subsequence(t, tuple(special_token_encoded), (i+256,))
    return t

def tuple_to_bytes(t, vocab):
    return b''.join([vocab[i] for i in t])

def is_subsequence(word, subseq, vocab):
    byte_subseq = tuple_to_bytes(subseq, vocab)
    return byte_subseq in word

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
        if lst[i:i+len(subsequence)] == subsequence:
            # Replace the subsequence with the replacement
            lst = lst[:i] + replacement + lst[i+len(subsequence):]
            i += len(replacement)  # Move past the replacement
        else:
            i += 1
    return lst

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # setup
    vocab = {i+256: encode(special_token) for i, special_token in enumerate(special_tokens)}
    vocab.update({i: bytes([i]) for i in range(256)})
    merges = []

    corpus = open(input_path).read()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenization = findall(PAT, corpus)
    pre_tokenization = [encode(word) for word in pre_tokenization]
    pre_tokenization = [word for word in pre_tokenization if len(word) > 1] # change to list of ints

    corpus = Counter(pre_tokenization)
    corpus = {word: {"freq": freq, "repr": tuple(word)} for word, freq in corpus.items()}
    byte_pairs = defaultdict(int)
    for word, word_info in corpus.items():
        freq, repr = word_info["freq"], word_info["repr"]
        for i in range(len(repr)-1):
            byte_pairs[repr[i:i+2]] += freq
    
    # BPE training
    while len(vocab) < vocab_size:
        new_vocab_idx = len(vocab)
        max_byte_pair, max_count = max(
            byte_pairs.items(),
            key=lambda item: (item[1], (vocab[item[0][0]], vocab[item[0][1]]))
        )
        vocab[new_vocab_idx] = tuple_to_bytes(max_byte_pair, vocab)
        del byte_pairs[max_byte_pair]
        for word, word_info in corpus.items():
            freq, repr = word_info["freq"], word_info["repr"]
            if is_subsequence(word, max_byte_pair, vocab):
                new_sequence = replace_subsequence(repr, max_byte_pair, (new_vocab_idx,))
                for i in range(len(new_sequence) - 1):
                    curr_byte_pair = new_sequence[i:i+2]
                    if new_vocab_idx in curr_byte_pair:
                        byte_pairs[curr_byte_pair] += freq
                    if curr_byte_pair[0] == new_vocab_idx and curr_byte_pair[1] == new_vocab_idx:
                        byte_pairs[(max_byte_pair[1], max_byte_pair[0])] -= freq
                    elif curr_byte_pair[0] == new_vocab_idx:
                        byte_pairs[(max_byte_pair[1], curr_byte_pair[1])] -= freq
                    elif curr_byte_pair[1] == new_vocab_idx:
                        byte_pairs[(curr_byte_pair[0], max_byte_pair[0])] -= freq
                corpus[word]["repr"] = new_sequence
        merges.append((vocab[max_byte_pair[0]], vocab[max_byte_pair[1]]))
    
    return vocab, merges


