"""
This module defines a class for deduplicating text using MinHash and MinHashLSH.

The file includes the following import statements:
- Standard libraries: importlib, uuid, functools, itertools, typing, and os
- Third-party libraries: nltk.tokenize, datasketch.MinHash, datasketch.MinHashLSH, asyncio, and diskcache.Index

The file also includes the following functions and classes:
- ngrams(sequence, n): Generates n-grams from a sequence.
- tokenized_on_words(texts): Tokenizes a list of texts into words using NLTK's word tokenizer.
- tokenize_on_ngrams(texts, n): Tokenizes a list of texts into n-grams.
- MinHashDedup: A class for deduplicating text using MinHash and MinHashLSH.

To use this module, you can import the MinHashDedup class and instantiate it with desired parameters. Then, use the filter_processing method to process input texts asynchronously and detect duplicates.

Example usage:
from minhash_dedup import MinHashDedup

deduper = MinHashDedup(num_perm=128, seed=1, tokenizer_type="words", n=5, threshold=0.9, storage="dict")
async for result in deduper.filter_processing(inputs):
    print(result)
"""

import importlib
import uuid
from functools import partial
from itertools import tee
from typing import (
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Dict,
    Any
)
from nltk.tokenize import word_tokenize
from datasketch import MinHash, MinHashLSH
import asyncio
from typing import AsyncIterator


# Copied from: https://github.com/huggingface/datatrove/blob/main/src/datatrove/utils/text.py#L89C1-L95C65
async def ngrams(sequence: Iterable[str], n: int) -> Iterator[Tuple[str, ...]]:
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


async def tokenized_on_words(texts: Iterable[str]) -> List[Set[bytes]]:
    """Tokenizes a list of texts into words asynchronously"""
    result = []
    for text in texts:
        # Allow other coroutines to run occasionally
        await asyncio.sleep(0)
        result.append({w.encode("utf-8") for w in word_tokenize(text)})
    return result


async def tokenize_on_ngrams(texts: Iterable[str], n: int = 1) -> List[Set[bytes]]:
    """Tokenizes a list of texts into ngrams asynchronously"""
    result = []
    for text in texts:
        # Allow other coroutines to run occasionally 
        await asyncio.sleep(0)
        ngram_iter = await ngrams(text, n=n)
        result.append({"".join(ngram).encode("utf-8") for ngram in ngram_iter})
    return result


class MiniHashDedup():
    """Deduplicates text using `MinHash` and `MinHashLSH`.

    `MiniHashDedup` can detect near-duplicates in datasets. The idea roughly translates
    to the following steps:
    1. Tokenize the text into words or ngrams.
    2. Create a `MinHash` for each text.
    3. Store the `MinHashes` in a `MinHashLSH`.
    4. Check if the `MinHash` is already in the `LSH`, if so, it is a duplicate.

    Attributes:
        num_perm: the number of permutations to use. Defaults to `128`.
        seed: the seed to use for the MinHash. This seed must be the same
            used for `MinHash`, keep in mind when both steps are created. Defaults to `1`.
        tokenizer: the tokenizer to use. Available ones are `words` or `ngrams`.
            If `words` is selected, it tokenize the text into words using nltk's
            word tokenizer. `ngram` estimates the ngrams (together with the size
            `n`) using. Defaults to `words`.
        n: the size of the ngrams to use. Only relevant if `tokenizer="ngrams"`. Defaults to `5`.
        threshold: the threshold to consider two MinHashes as duplicates.
            Values closer to 0 detect more duplicates. Defaults to `0.9`.
        storage: the storage to use for the LSH. Can be `dict` to store the index
            in memory, or `disk`. Keep in mind, `disk` is an experimental feature
            not defined in `datasketch`, that is based on DiskCache's `Index` class.
            It should work as a `dict`, but backed by disk, but depending on the system
            it can be slower. Defaults to `dict`.
            which uses a custom `shelve` backend. Note the `disk`
            is an experimetal feature that may cause issues. Defaults to `dict`.

    Input columns:
        - text (`str`): the texts to be filtered.

    Output columns:
        - keep_row_after_minhash_filtering (`bool`): boolean indicating if the piece `text` is
            not a duplicate i.e. this text should be kept.

    Categories:
        - filtering
    References:
        - [`datasketch documentation`](https://ekzhu.github.io/datasketch/lsh.html)
        - [Identifying and Filtering Near-Duplicate Documents](https://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf)
        - [Diskcache's Index](https://grantjenks.com/docs/diskcache/api.html#diskcache.Index)
    Examples:
        from datasketch import MinHash, MinHashLSH
        set1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
                    'estimating', 'the', 'similarity', 'between', 'datasets'])
        set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
                    'estimating', 'the', 'similarity', 'between', 'documents'])
        set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for',
                    'estimating', 'the', 'similarity', 'between', 'documents'])
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        m3 = MinHash(num_perm=128)
        for d in set1:
            m1.update(d.encode('utf8'))
        for d in set2:
            m2.update(d.encode('utf8'))
        for d in set3:
            m3.update(d.encode('utf8'))
        # Create LSH index
        lsh = MinHashLSH(threshold=0.5, num_perm=128)
        lsh.insert("m2", m2)
        lsh.insert("m3", m3)
        result = lsh.query(m1)
        print("Approximate neighbours with Jaccard similarity > 0.5", result)
    """

    def __init__(self,
                 num_perm: Optional[int] = 128,
                 seed: Optional[int] = 1,
                 tokenizer_type: Literal["words", "ngrams"] = "words",
                 n: Optional[int] = 5,
                 threshold: Optional[float] = 0.9,
                 storage: Literal["dict", "disk"] = "dict"
                 ):
        self.num_perm = num_perm
        self.seed = seed
        self.tokenizer_type = tokenizer_type
        self.n = n
        self.threshold = threshold
        self.storage = storage
        # self.tokenizer_init()
        self.hasher = MinHash.bulk
        self.lsh = MinHashLSH(num_perm=self.num_perm, threshold=self.threshold, storage_config={"type": self.storage})
        if not 0 < threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")

        # asyncio.create_task(self.tokenizer_init())

    async def tokenizer_init(self) -> None:
        if self.tokenizer_type == "words":
            if not importlib.import_module("nltk"):
                raise ImportError(
                    "`nltk` is needed to tokenize based on words, but is not installed. "
                    "Please install it using `pip install nltk`. Then run `nltk.download('punkt_tab')`."
                )
            self._tokenizer = tokenized_on_words
        else:
            self._tokenizer = partial(tokenize_on_ngrams, n=self.n)

    async def clear(self):
        """Clear the LSH index"""
        self.lsh = MinHashLSH(num_perm=self.num_perm,
                              threshold=self.threshold,
                              storage_config={"type": self.storage})

    async def filter_processing(self,
                                inputs: List[Dict[str, Any]],
                                need_return_dup_lines: bool = True,
                                batch_size: Optional[int] = 1000) -> AsyncIterator[Dict[str, Any]]:
        """Process input texts asynchronously to detect duplicates.
        
        Args:
            inputs: List of dictionaries containing text to process
            need_return_dup_lines: Whether to yield duplicate entries
            batch_size: Size of batches to process
            
        Yields:
            Dictionary with original input and duplicate status
        """
        await self.tokenizer_init()  # wait the tokenizer init

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]

            # Pre-tokenize all texts asynchronously
            tokenized_texts = []
            for input_dict in batch:
                tokens = await self._tokenizer([input_dict["input"]])
                tokenized_texts.append(tokens[0])
                await asyncio.sleep(0)

            # Generate MinHashes for all texts at once
            mini_hashes = self.hasher(tokenized_texts, num_perm=self.num_perm, seed=self.seed)

            # Process each text
            for idx, (input_dict, minhash) in enumerate(zip(batch, mini_hashes)):
                await asyncio.sleep(0)

                is_duplicate = bool(self.lsh.query(minhash))
                input_dict["is_duplicated_by_minhash_filtering"] = is_duplicate

                if not is_duplicate:
                    self.lsh.insert(str(uuid.uuid4()), minhash)

                if not is_duplicate or need_return_dup_lines:
                    yield input_dict
