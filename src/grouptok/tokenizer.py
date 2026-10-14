"""GroupedTokenizer: a byte-level BPE shared by two or more languages, plus a Grouping of its vocabulary."""
from __future__ import annotations

import json
import operator
import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import reduce
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .align import Aligner, AlignerConfig, align_pairs
from .grouping import Grouping, GroupingConfig, build_groups
from .source import Source

PathLike = str | os.PathLike
Pair = tuple[str, str]

DEFAULT_SPECIAL_TOKENS: tuple[str, str, str] = ('<|endoftext|>', '<|im_start|>', '<|im_end|>')  # pad, bos, eos -> ids 0, 1, 2
# the files of a tokenizer folder that save_pretrained copies (transformers' names), leaving out model weights and the like
TOKENIZER_FILES = ('*token*', 'vocab*', 'merges.txt', '*.model', 'chat_template*')


@dataclass(frozen=True)
class TokenizerConfig:
    """Settings for training a grouped tokenizer.

    Attributes:
        vocab_size: Size of the BPE vocabulary, reserved tokens included; ignored when `train` gets a pretrained tokenizer.
        special_tokens: The `(pad, bos, eos)` tokens. They get the first ids of a trained BPE (0, 1 and 2), and
            `decode(skip_special=True)` drops them. They may repeat, e.g. `("<|endoftext|>",) * 3` for GPT-2's tokenizer.
        additional_tokens: More reserved tokens after the special ones (e.g. `'<think>'`). They are never grouped and
            are kept when decoding.
        chat_template: Jinja chat template saved to `tokenizer_config.json` for transformers' `apply_chat_template`.
        model_max_length: Maximum sequence length recorded in `tokenizer_config.json`.
        grouping: How alignment links become groups.
        aligner: The neural aligner and how links are extracted.

    Raises:
        ValueError: If `special_tokens` is not three tokens, or an additional token appears twice or is also a special
            token.

    Examples:
        >>> from grouptok import GroupingConfig, TokenizerConfig
        >>> config = TokenizerConfig(vocab_size=32000, additional_tokens=("<think>", "</think>"),
        ...                          grouping=GroupingConfig(max_group_size=4))
        >>> config.special_tokens
        ('<|endoftext|>', '<|im_start|>', '<|im_end|>')
        >>> TokenizerConfig(special_tokens=("<pad>", "<s>"))
        Traceback (most recent call last):
        ...
        ValueError: special_tokens must be (pad, bos, eos)
    """
    vocab_size: int = 8192
    special_tokens: tuple[str, str, str] = DEFAULT_SPECIAL_TOKENS
    additional_tokens: tuple[str, ...] = ()
    chat_template: str | None = None
    model_max_length: int = 131072
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    aligner: AlignerConfig = field(default_factory=AlignerConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'special_tokens', tuple(self.special_tokens))
        object.__setattr__(self, 'additional_tokens', tuple(self.additional_tokens))
        if len(self.special_tokens) != 3:
            raise ValueError('special_tokens must be (pad, bos, eos)')
        if len(set(self.additional_tokens)) != len(self.additional_tokens) or set(self.additional_tokens) & set(self.special_tokens):
            raise ValueError('additional tokens must be distinct and not special tokens')


def train_bpe(texts: Iterable[str], vocab_size: int, reserved_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS, *,
              progress: bool = True) -> Tokenizer:
    """Byte-level BPE (no prefix space) with the reserved tokens as its first ids"""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=progress, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=list(reserved_tokens))
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


class GroupedTokenizer:
    """A byte-level BPE tokenizer whose tokens are grouped by translation.

    Tokens of a group translate each other, and every token id `t` is the pair `grouping.pair(t) = (group, member)`.
    Encoding and decoding work on ordinary token ids, so the tokenizer can be used anywhere a BPE tokenizer is; models
    read the grouping to share parameters within groups.

    Attributes:
        tokenizer: The underlying `tokenizers.Tokenizer`.
        grouping: The `(group, member)` pair of every token.
        special_tokens: The `(pad, bos, eos)` tokens.
        additional_tokens: More reserved tokens, kept when decoding.
        chat_template: Jinja chat template for transformers, or `None`.
        model_max_length: Maximum sequence length recorded for transformers.
        source: The folder whose tokenizer files `save_pretrained` copies as they are, or `None` to write them.

    Examples:
        >>> from grouptok import GroupedTokenizer
        >>> tok = GroupedTokenizer.from_pretrained("my-tokenizer")
        >>> tok.pairs(tok.encode(" king roi"))   # a model predicts the group, then the member
        [(977, 0), (977, 1)]
        >>> tok.group_of(" king")
        [' king', ' roi']
    """

    def __init__(self, tokenizer: Tokenizer, grouping: Grouping, *, special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
                 additional_tokens: Sequence[str] = (), chat_template: str | None = None, model_max_length: int = 131072,
                 source: PathLike | None = None) -> None:
        """Combine a trained BPE tokenizer with a grouping of its vocabulary.

        Most code gets a tokenizer from `train` or `from_pretrained` instead.

        Args:
            tokenizer: A byte-level BPE tokenizer.
            grouping: A grouping of exactly the tokenizer's vocabulary.
            special_tokens: The `(pad, bos, eos)` tokens; all must be in the vocabulary.
            additional_tokens: More reserved tokens; all must be in the vocabulary.
            chat_template: Jinja chat template for transformers, or `None`.
            model_max_length: Maximum sequence length recorded for transformers.
            source: A folder holding the tokenizer's files (`tokenizer.json`, ...). `save_pretrained` copies them as
                they are instead of writing `tokenizer.json` and `tokenizer_config.json`.

        Raises:
            ValueError: If the grouping covers a different number of tokens than the tokenizer has, or a special or
                additional token is missing from the vocabulary.

        Examples:
            An ungrouped baseline, every token a group of its own:

            >>> from tokenizers import Tokenizer
            >>> from grouptok import GroupedTokenizer, Grouping
            >>> bpe = Tokenizer.from_file("my-tokenizer/tokenizer.json")
            >>> flat = GroupedTokenizer(bpe, Grouping.flat(bpe.get_vocab_size()))
            >>> flat.group_of(" king")
            [' king']
        """
        if grouping.vocab_size != tokenizer.get_vocab_size():
            raise ValueError(f'grouping covers {grouping.vocab_size} tokens but the tokenizer has {tokenizer.get_vocab_size()}')
        self.tokenizer = tokenizer
        self.grouping = grouping
        self.special_tokens = tuple(special_tokens)
        self.additional_tokens = tuple(additional_tokens)
        self.chat_template = chat_template
        self.model_max_length = model_max_length
        self.source = source
        missing = [t for t in self.special_tokens + self.additional_tokens if tokenizer.token_to_id(t) is None]
        if missing:
            raise ValueError(f'tokens missing from the vocabulary: {missing}')

    # ---- training
    @classmethod
    def train(cls, pairs: Iterable[Pair | Source], config: TokenizerConfig = TokenizerConfig(), *,
              tokenizer: PathLike | None = None, aligner: Aligner | None = None, progress: bool = True) -> GroupedTokenizer:
        """Train a grouped tokenizer on sentence pairs.

        Trains a BPE on both sides of the pairs, aligns the subwords of each pair, then groups subwords that are often
        aligned. For more than two languages, pass the pairs of every language pair together: links are pooled, and a
        group can gather words of all of them. Training needs `pip install "grouptok[train]"`.

        Sources are read first. Pairs are aligned in one pass per aligner, and the link counts of all passes are pooled
        before grouping. Tokens that are only part of a UTF-8 character (they decode to `�`) are never grouped.

        Args:
            pairs: `(text, translation)` pairs and/or `Source`s to read them from. Pairs with an empty side are skipped.
                The first side of each pair supplies member 0 of the groups it creates.
            config: Vocabulary size, reserved tokens, grouping and aligner settings.
            tokenizer: A local folder with a pretrained tokenizer's files, `tokenizer.json` among them, to group instead
                of training a BPE. Its vocabulary and ids are kept and `save_pretrained` copies its files as they are;
                `config.vocab_size` is ignored, and the special and additional tokens must already be in its vocabulary.
            aligner: A custom aligner for the pairs and the sources without their own; defaults to
                `HFAligner(config.aligner)`.
            progress: Print progress while training.

        Returns:
            The trained tokenizer.

        Raises:
            ValueError: If no pair has text on both sides, a source has no rows, or a special or additional token is
                missing from `tokenizer`.

        Examples:
            >>> from grouptok import GroupedTokenizer, TokenizerConfig
            >>> pairs = [("The king said unto the people", "Le roi dit au peuple")]   # thousands of pairs in practice
            >>> tok = GroupedTokenizer.train(pairs, TokenizerConfig(vocab_size=8192))
            >>> tok.save_pretrained("my-tokenizer")

            For more than two languages, pool the pairs of every language pair:

            >>> tok = GroupedTokenizer.train(en_fr_pairs + en_de_pairs + fr_de_pairs)

            Or describe each dataset as a `Source`:

            >>> from grouptok import Source
            >>> tok = GroupedTokenizer.train([Source("en-fr.jsonl", columns=("en", "fr"), rows=100_000),
            ...                               Source("en-de.jsonl", columns=("en", "de"))])

            Or mix pairs and sources:

            >>> tok = GroupedTokenizer.train(pairs + [Source("en-de.jsonl", columns=("en", "de"))])

            Or group a pretrained tokenizer's vocabulary, from the Hugging Face Hub or any local folder:

            >>> from huggingface_hub import snapshot_download
            >>> qwen = snapshot_download("Qwen/Qwen2.5-0.5B", allow_patterns=["*token*"])   # the tokenizer files only
            >>> tok = GroupedTokenizer.train(pairs, tokenizer=qwen)
            >>> gpt2 = snapshot_download("openai-community/gpt2", allow_patterns=["*token*"])
            >>> tok = GroupedTokenizer.train(pairs, TokenizerConfig(special_tokens=("<|endoftext|>",) * 3), tokenizer=gpt2)
        """
        reserved = config.special_tokens + config.additional_tokens
        bpe = None if tokenizer is None else Tokenizer.from_file(os.path.join(tokenizer, 'tokenizer.json'))
        if bpe is not None and (missing := [t for t in reserved if bpe.token_to_id(t) is None]):
            raise ValueError(f'tokens missing from the vocabulary: {missing}')   # before reading sources and aligning
        items = list(pairs)
        plain = [(a, b) for a, b in (p for p in items if not isinstance(p, Source)) if a and b]
        by_aligner: dict[AlignerConfig | None, list[Pair]] = {None: plain} if plain else {}   # None: the default aligner
        for source in (p for p in items if isinstance(p, Source)):
            rows = source.load()
            if progress:
                print(f'{len(rows)} pairs from {source.path}' + (f' ({source.subset})' if source.subset else ''))
            by_aligner.setdefault(source.aligner, []).extend(rows)
        if not by_aligner:
            raise ValueError('no sentence pairs with text on both sides')
        if bpe is None:
            bpe = train_bpe((text for rows in by_aligner.values() for pair in rows for text in pair), config.vocab_size, reserved,
                            progress=progress)
        passes = (align_pairs(rows, bpe, source_aligner or config.aligner, aligner=None if source_aligner else aligner,
                              progress=progress) for source_aligner, rows in by_aligner.items())   # one encoder loaded at a time
        links = reduce(operator.add, passes)
        fragments = [t for t, text in enumerate(bpe.decode_batch([[t] for t in range(bpe.get_vocab_size())])) if '�' in text]
        grouping = build_groups(links, config.grouping, reserved=[bpe.token_to_id(t) for t in reserved] + fragments)
        tok = cls(bpe, grouping, special_tokens=config.special_tokens, additional_tokens=config.additional_tokens,
                  chat_template=config.chat_template, model_max_length=config.model_max_length, source=tokenizer)
        if progress:
            print(f'{grouping.num_groups} groups for {grouping.vocab_size} tokens; group sizes: {grouping.group_sizes()}')
        return tok

    # ---- persistence: tokenizer.json, tokenizer_config.json (loadable with transformers' AutoTokenizer) and groups.json
    @classmethod
    def from_pretrained(cls, path: PathLike) -> GroupedTokenizer:
        """Load a tokenizer saved by `save_pretrained`.

        `save_pretrained` on the loaded tokenizer copies the directory's tokenizer files as they are.

        Args:
            path: A local directory with `tokenizer.json` and `groups.json`. Directories saved by older grouptok
                versions also need `tokenizer_config.json`, where they keep the reserved tokens.

        Returns:
            The loaded tokenizer.

        Raises:
            FileNotFoundError: If one of the files is missing.
            ValueError: If the files don't describe the same vocabulary.

        Examples:
            >>> tok = GroupedTokenizer.from_pretrained("my-tokenizer")

            From the Hugging Face Hub, download the repository first:

            >>> from huggingface_hub import snapshot_download
            >>> tok = GroupedTokenizer.from_pretrained(snapshot_download("your-name/my-tokenizer"))
        """
        tokenizer =Tokenizer.from_file(os.path.join(path, 'tokenizer.json'))
        config_path = os.path.join(path, 'tokenizer_config.json')
        config: dict[str, Any] = {}
        if os.path.exists(config_path):   # a pretrained tokenizer's folder may have none
            with open(config_path, encoding='utf-8') as f:
                config = json.load(f)
        with open(os.path.join(path, 'groups.json'), encoding='utf-8') as f:
            groups = json.load(f)
        if 'special_tokens' in groups:   # a pretrained tokenizer's config doesn't name grouptok's pad / bos / eos
            special, additional = groups['special_tokens'], groups['additional_tokens']
        else:   # saved before groups.json listed the reserved tokens
            special = (config['pad_token'], config['bos_token'], config['eos_token'])
            added = sorted(config.get('added_tokens_decoder', {}).items(), key=lambda item: int(item[0]))
            additional = [info['content'] for _, info in added if info['content'] not in special]
        return cls(tokenizer, Grouping.load(os.path.join(path, 'groups.json')), special_tokens=special, additional_tokens=additional,
                   chat_template=config.get('chat_template'), model_max_length=config.get('model_max_length', 131072), source=path)

    def _tokenizer_data(self) -> dict[str, Any]:
        """tokenizer.json content, with only pad / bos / eos flagged special (additional tokens stay visible when decoding)"""
        data = json.loads(self.tokenizer.to_str())
        for token in data.get('added_tokens', []):
            token['special'] = token['content'] in self.special_tokens
        return data

    def save_pretrained(self, path: PathLike) -> None:
        """Save the tokenizer to a directory.

        Writes `groups.json` (the grouping and the reserved tokens). A tokenizer that came from a folder (a pretrained
        one, or one loaded by `from_pretrained`) gets that folder's tokenizer files copied as they are; a tokenizer
        trained by grouptok gets `tokenizer.json` (a standard `tokenizers` file) and `tokenizer_config.json` (so
        transformers' `AutoTokenizer.from_pretrained` can load the directory).

        Args:
            path: The directory, created if it doesn't exist.

        Examples:
            >>> tok.save_pretrained("my-tokenizer")

            transformers loads the same directory (the BPE only, without the grouping):

            >>> from transformers import AutoTokenizer
            >>> hf = AutoTokenizer.from_pretrained("my-tokenizer")
            >>> hf("The king", add_special_tokens=False).input_ids == tok.encode("The king")
            True
        """
        os.makedirs(path, exist_ok=True)
        if self.source is None:
            with open(os.path.join(path, 'tokenizer.json'), 'w', encoding='utf-8') as f:
                json.dump(self._tokenizer_data(), f, ensure_ascii=False, indent=2)
            with open(os.path.join(path, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
                json.dump(self._hf_config(), f, ensure_ascii=False, indent=4)
        elif not os.path.samefile(self.source, path):
            for name in os.listdir(self.source):
                file = os.path.join(self.source, name)
                if os.path.isfile(file) and any(fnmatch(name, pattern) for pattern in TOKENIZER_FILES):
                    shutil.copy(file, path)
        self.grouping.save(os.path.join(path, 'groups.json'), tokens=[self.tokenizer.id_to_token(t) for t in range(self.vocab_size)],
                           special_tokens=list(self.special_tokens), additional_tokens=list(self.additional_tokens))

    def _hf_config(self) -> dict[str, Any]:
        pad, bos, eos = self.special_tokens
        config = {
            'add_bos_token': False,
            'add_eos_token': False,
            'add_prefix_space': False,
            'added_tokens_decoder': {str(self.tokenizer.token_to_id(t)): {'content': t, 'lstrip': False, 'normalized': False, 'rstrip': False,
                                                                          'single_word': False, 'special': t in self.special_tokens}
                                     for t in self.special_tokens + self.additional_tokens},
            'additional_special_tokens': [bos, eos],
            'bos_token': bos,
            'clean_up_tokenization_spaces': False,
            'eos_token': eos,
            'legacy': True,
            'model_max_length': self.model_max_length,
            'pad_token': pad,
            'sp_model_kwargs': {},
            'spaces_between_special_tokens': False,
            'unk_token': pad,
            'chat_template': self.chat_template,
            'tokenizer_class': 'PreTrainedTokenizerFast',
        }
        if self.chat_template is None:
            del config['chat_template']
        return config

    # ---- vocabulary
    @property
    def vocab_size(self) -> int:
        """The number of tokens in the vocabulary.

        Examples:
            >>> tok.vocab_size == len(tok)
            True
        """
        return self.tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def pad_id(self) -> int:
        """The id of the padding token (0 for a trained tokenizer).

        Examples:
            >>> tok.pad_id
            0
        """
        return self.tokenizer.token_to_id(self.special_tokens[0])

    @property
    def bos_id(self) -> int:
        """The id of the beginning-of-sequence token (1 for a trained tokenizer).

        Examples:
            >>> tok.bos_id
            1
        """
        return self.tokenizer.token_to_id(self.special_tokens[1])

    @property
    def eos_id(self) -> int:
        """The id of the end-of-sequence token (2 for a trained tokenizer).

        Examples:
            >>> tok.eos_id
            2
        """
        return self.tokenizer.token_to_id(self.special_tokens[2])

    def token_to_id(self, token: str) -> int | None:
        """Look up the id of a token.

        Args:
            token: A token as stored in the vocabulary, where `Ġ` marks a leading space (e.g. `'Ġking'`).

        Returns:
            The token's id, or `None` if it is not in the vocabulary.

        Examples:
            >>> tok.token_to_id("Ġking") == tok.encode(" king")[0]
            True
            >>> tok.token_to_id(" king") is None   # the vocabulary writes the space as Ġ
            True
        """
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        """Look up the token of an id.

        Args:
            token_id: A token id.

        Returns:
            The token as stored in the vocabulary (e.g. `'Ġking'`), or `None` if the id is out of range.

        Examples:
            >>> tok.id_to_token(tok.eos_id)
            '<|im_end|>'
        """
        return self.tokenizer.id_to_token(token_id)

    # ---- text <-> ids
    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        """Encode a text into token ids.

        Args:
            text: The text to encode.
            bos: Prepend the beginning-of-sequence token.
            eos: Append the end-of-sequence token.

        Returns:
            The token ids.

        Examples:
            >>> ids = tok.encode("The king", bos=True, eos=True)
            >>> ids[0] == tok.bos_id and ids[-1] == tok.eos_id
            True
            >>> tok.decode(ids)
            'The king'
        """
        return [self.bos_id] * bos + self.tokenizer.encode(text, add_special_tokens=False).ids + [self.eos_id] * eos

    def encode_batch(self, texts: Sequence[str], *, bos: bool = False, eos: bool = False) -> list[list[int]]:
        """Encode several texts in parallel.

        Args:
            texts: The texts to encode.
            bos: Prepend the beginning-of-sequence token to each text.
            eos: Append the end-of-sequence token to each text.

        Returns:
            The token ids of each text, in order.

        Examples:
            >>> tok.encode_batch(["The king", "Le roi"]) == [tok.encode("The king"), tok.encode("Le roi")]
            True
        """
        return [[self.bos_id] * bos + e.ids + [self.eos_id] * eos for e in self.tokenizer.encode_batch(list(texts), add_special_tokens=False)]

    def tokenize(self, text: str) -> list[str]:
        """Split a text into tokens.

        Args:
            text: The text to split.

        Returns:
            The tokens as stored in the vocabulary, where `Ġ` marks a leading space.

        Examples:
            >>> tok.tokenize("The king")
            ['The', 'Ġking']
        """
        return self.tokenizer.encode(text, add_special_tokens=False).tokens

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        """Decode token ids back into text.

        Args:
            ids: The token ids.
            skip_special: Drop the pad, bos and eos tokens. Additional tokens are always kept.

        Returns:
            The decoded text.

        Examples:
            >>> ids = tok.encode("The king", eos=True)
            >>> tok.decode(ids)
            'The king'
            >>> tok.decode(ids, skip_special=False)
            'The king<|im_end|>'
        """
        if skip_special:
            special = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special]
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def pairs(self, ids: Sequence[int]) -> list[tuple[int, int]]:
        """Map token ids to their `(group, member)` pairs.

        Args:
            ids: The token ids.

        Returns:
            The `(group, member)` pair of each id, in order.

        Examples:
            >>> tok.pairs(tok.encode(" king roi"))
            [(977, 0), (977, 1)]
        """
        return [self.grouping.pair(i) for i in ids]

    # ---- groups
    def _token_id(self, token: str | int) -> int:
        if isinstance(token, int):
            return token
        token_id = self.tokenizer.token_to_id(token)
        if token_id is not None:
            return token_id
        ids = self.encode(token)
        if len(ids) != 1:
            raise KeyError(f'{token!r} is {len(ids)} tokens, not one; pass a token, an id, or text that encodes to one token '
                           f'(word-initial tokens include the leading space: {" " + token.strip()!r})')
        return ids[0]

    def group_of(self, token: str | int) -> list[str]:
        """List the members of a token's group as text.

        Args:
            token: A token id, a token as stored in the vocabulary (`'Ġpark'`), or text that encodes to a single token
                (`' park'`; word-initial tokens include the leading space).

        Returns:
            The members of the token's group as text, in member order, the token included.

        Raises:
            KeyError: If `token` is text that encodes to more than one token.

        Examples:
            >>> tok.group_of(" king")
            [' king', ' roi']
            >>> tok.group_of("Ġking") == tok.group_of(tok.token_to_id("Ġking"))
            True
        """
        return [self.tokenizer.decode([t]) for t in self.grouping.members(self._token_id(token))]

    def readable_groups(self, min_size: int = 2) -> list[list[str]]:
        """List the groups as text.

        Args:
            min_size: Leave out groups with fewer members. The default leaves out tokens without a translation.

        Returns:
            The members of each group as text, in member order.

        Examples:
            >>> tok.readable_groups()[:2]
            [[' house', ' maison'], [' king', ' roi']]
        """
        return [[self.tokenizer.decode([t]) for t in members] for members in self.grouping.groups() if len(members) >= min_size]
