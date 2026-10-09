"""GroupedTokenizer: a byte-level BPE shared by two or more languages, plus a Grouping of its vocabulary."""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .align import Aligner, AlignerConfig, align_pairs
from .grouping import Grouping, GroupingConfig, build_groups

PathLike = str | os.PathLike
Pair = tuple[str, str]

DEFAULT_SPECIAL_TOKENS: tuple[str, str, str] = ('<|endoftext|>', '<|im_start|>', '<|im_end|>')  # pad, bos, eos -> ids 0, 1, 2


@dataclass(frozen=True)
class TokenizerConfig:
    """Everything that defines a trained grouped tokenizer"""
    vocab_size: int = 8192
    special_tokens: tuple[str, str, str] = DEFAULT_SPECIAL_TOKENS   # (pad, bos, eos); ids 0, 1, 2; removed by decode(skip_special=True)
    additional_tokens: tuple[str, ...] = ()                         # more reserved tokens after them (e.g. '<think>'); kept when decoding
    chat_template: str | None = None                                # Jinja chat template for transformers' apply_chat_template
    model_max_length: int = 131072
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    aligner: AlignerConfig = field(default_factory=AlignerConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'special_tokens', tuple(self.special_tokens))
        object.__setattr__(self, 'additional_tokens', tuple(self.additional_tokens))
        if len(self.special_tokens) != 3:
            raise ValueError('special_tokens must be (pad, bos, eos)')
        reserved = self.special_tokens + self.additional_tokens
        if len(set(reserved)) != len(reserved):
            raise ValueError('special and additional tokens must be distinct')


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
    """A byte-level BPE whose tokens are grouped by translation: tokens of a group translate each other.

    Every token id t is the pair grouping.pair(t) = (group, member). Encoding and decoding work on ordinary token ids,
    so the tokenizer can be used anywhere a BPE tokenizer is; models read the grouping to share parameters within groups."""

    def __init__(self, tokenizer: Tokenizer, grouping: Grouping, *, special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
                 additional_tokens: Sequence[str] = (), chat_template: str | None = None, model_max_length: int = 131072) -> None:
        if grouping.vocab_size != tokenizer.get_vocab_size():
            raise ValueError(f'grouping covers {grouping.vocab_size} tokens but the tokenizer has {tokenizer.get_vocab_size()}')
        self.tokenizer = tokenizer
        self.grouping = grouping
        self.special_tokens = tuple(special_tokens)
        self.additional_tokens = tuple(additional_tokens)
        self.chat_template = chat_template
        self.model_max_length = model_max_length
        missing = [t for t in self.special_tokens + self.additional_tokens if tokenizer.token_to_id(t) is None]
        if missing:
            raise ValueError(f'tokens missing from the vocabulary: {missing}')

    # ---- training
    @classmethod
    def train(cls, pairs: Iterable[Pair], config: TokenizerConfig = TokenizerConfig(), *, aligner: Aligner | None = None,
              progress: bool = True) -> GroupedTokenizer:
        """Train on sentence pairs: BPE on both sides, subword alignment, then grouping.

        Pairs are (first-language text, second-language text); pairs with an empty side are skipped. For more than two
        languages, pass the pairs of every language pair together: links are pooled, and a group can gather words of all
        of them. The first side of each pair supplies member 0 of the groups it creates. The aligner defaults to
        HFAligner(config.aligner) (pip install grouptok[train])."""
        pairs = [(a, b) for a, b in pairs if a and b]
        if not pairs:
            raise ValueError('no sentence pairs with text on both sides')
        reserved = config.special_tokens + config.additional_tokens
        bpe = train_bpe((text for pair in pairs for text in pair), config.vocab_size, reserved, progress=progress)
        links = align_pairs(pairs, bpe, config.aligner, aligner=aligner, progress=progress)
        grouping = build_groups(links, config.grouping, reserved=[bpe.token_to_id(t) for t in reserved])
        tok = cls(bpe, grouping, special_tokens=config.special_tokens, additional_tokens=config.additional_tokens,
                  chat_template=config.chat_template, model_max_length=config.model_max_length)
        if progress:
            print(f'{grouping.num_groups} groups for {grouping.vocab_size} tokens; group sizes: {grouping.group_sizes()}')
        return tok

    # ---- persistence: tokenizer.json, tokenizer_config.json (loadable with transformers' AutoTokenizer) and groups.json
    @classmethod
    def from_pretrained(cls, path: PathLike) -> GroupedTokenizer:
        tokenizer = Tokenizer.from_file(os.path.join(path, 'tokenizer.json'))
        with open(os.path.join(path, 'tokenizer_config.json'), encoding='utf-8') as f:
            config = json.load(f)
        special = (config['pad_token'], config['bos_token'], config['eos_token'])
        added = sorted(config.get('added_tokens_decoder', {}).items(), key=lambda item: int(item[0]))
        additional = [info['content'] for _, info in added if info['content'] not in special]
        return cls(tokenizer, Grouping.load(os.path.join(path, 'groups.json')), special_tokens=special, additional_tokens=additional,
                   chat_template=config.get('chat_template'), model_max_length=config.get('model_max_length', 131072))

    def _tokenizer_data(self) -> dict[str, Any]:
        """tokenizer.json content, with only pad / bos / eos flagged special (additional tokens stay visible when decoding)"""
        data = json.loads(self.tokenizer.to_str())
        for token in data.get('added_tokens', []):
            token['special'] = token['content'] in self.special_tokens
        return data

    def save_pretrained(self, path: PathLike) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'tokenizer.json'), 'w', encoding='utf-8') as f:
            json.dump(self._tokenizer_data(), f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
            json.dump(self._hf_config(), f, ensure_ascii=False, indent=4)
        self.grouping.save(os.path.join(path, 'groups.json'), tokens=[self.tokenizer.id_to_token(t) for t in range(self.vocab_size)])

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
        return self.tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(self.special_tokens[0])

    @property
    def bos_id(self) -> int:
        return self.tokenizer.token_to_id(self.special_tokens[1])

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id(self.special_tokens[2])

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self.tokenizer.id_to_token(token_id)

    # ---- text <-> ids
    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        return [self.bos_id] * bos + self.tokenizer.encode(text, add_special_tokens=False).ids + [self.eos_id] * eos

    def encode_batch(self, texts: Sequence[str], *, bos: bool = False, eos: bool = False) -> list[list[int]]:
        return [[self.bos_id] * bos + e.ids + [self.eos_id] * eos for e in self.tokenizer.encode_batch(list(texts), add_special_tokens=False)]

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.encode(text, add_special_tokens=False).tokens

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        if skip_special:
            special = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special]
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def pairs(self, ids: Sequence[int]) -> list[tuple[int, int]]:
        """(group, member) of each token id"""
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
        """The members of a token's group as text, in member order. token: an id, a token string, or text that encodes
        to a single token (e.g. ' park' for the word-initial token)"""
        return [self.tokenizer.decode([t]) for t in self.grouping.members(self._token_id(token))]

    def readable_groups(self, min_size: int = 2) -> list[list[str]]:
        """Every group with at least min_size members, as text"""
        return [[self.tokenizer.decode([t]) for t in members] for members in self.grouping.groups() if len(members) >= min_size]
