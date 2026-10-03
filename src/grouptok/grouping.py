"""Groupings: a lossless (group, member) code for every token of a vocabulary, and how translation links become one."""
from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .align import LinkCounts

PathLike = str | os.PathLike


@dataclass(frozen=True)
class GroupingConfig:
    """How alignment links become groups"""
    max_group_size: int = 8     # members per group at most (the model's member head has this many outputs)
    min_link_count: int = 5     # a subword pair must be aligned at least this often to be grouped
    min_dice: float = 0.1       # and have at least this Dice score

    def __post_init__(self) -> None:
        if self.max_group_size < 2:
            raise ValueError('max_group_size must be at least 2')
        if self.min_link_count < 1:
            raise ValueError('min_link_count must be at least 1')
        if not 0 <= self.min_dice <= 1:
            raise ValueError('min_dice must be between 0 and 1')


@dataclass(frozen=True)
class Grouping:
    """Every token t is the pair (token_group[t], token_member[t]); the mapping is one-to-one.

    Members of a group are numbered 0, 1, ... in the order they joined it, and no group has more than
    max_group_size members. Tokens without a translation are groups of their own (member 0)."""
    token_group: tuple[int, ...]
    token_member: tuple[int, ...]
    max_group_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'token_group', tuple(self.token_group))
        object.__setattr__(self, 'token_member', tuple(self.token_member))
        if len(self.token_group) != len(self.token_member):
            raise ValueError('token_group and token_member must have the same length')
        if self.token_group and (min(self.token_group) < 0 or set(self.token_group) != set(range(max(self.token_group) + 1))):
            raise ValueError('group ids must be 0, 1, ..., number of groups - 1')
        if any(not 0 <= m < self.max_group_size for m in self.token_member):
            raise ValueError(f'member ids must be between 0 and max_group_size - 1 = {self.max_group_size - 1}')
        if len(set(zip(self.token_group, self.token_member))) != len(self.token_group):
            raise ValueError('two tokens have the same (group, member) pair')

    # ---- construction
    @classmethod
    def flat(cls, vocab_size: int) -> Grouping:
        """Every token its own group"""
        return cls(tuple(range(vocab_size)), (0,) * vocab_size, 1)

    @classmethod
    def from_groups(cls, groups: Sequence[Sequence[int]], max_group_size: int | None = None) -> Grouping:
        """Groups as lists of token ids in member order; together they must cover tokens 0 .. vocab size - 1 exactly once"""
        vocab_size = sum(map(len, groups))
        token_group, token_member = [-1] * vocab_size, [0] * vocab_size
        for g, members in enumerate(groups):
            for m, t in enumerate(members):
                if not 0 <= t < vocab_size or token_group[t] != -1:
                    raise ValueError(f'token {t} is out of range or in more than one group')
                token_group[t], token_member[t] = g, m
        return cls(tuple(token_group), tuple(token_member), max_group_size or max(map(len, groups), default=1))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Grouping:
        """From the content of groups.json; other keys are ignored"""
        return cls(tuple(data['token_group']), tuple(data['token_member']), int(data['max_group_size']))

    def to_dict(self, tokens: Sequence[str] | None = None) -> dict[str, Any]:
        """The content of groups.json; with the vocabulary's token strings, also a readable map of the groups with several members"""
        data: dict[str, Any] = {'max_group_size': self.max_group_size, 'token_group': list(self.token_group),
                                'token_member': list(self.token_member)}
        if tokens is not None:
            data['groups'] = {g: [tokens[t] for t in members] for g, members in enumerate(self.groups()) if len(members) > 1}
        return data

    @classmethod
    def load(cls, path: PathLike) -> Grouping:
        """Read groups.json (a file, or a directory containing it)"""
        path = os.path.join(path, 'groups.json') if os.path.isdir(path) else path
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def save(self, path: PathLike, tokens: Sequence[str] | None = None) -> None:
        """Write groups.json (a file, or a directory to write it into)"""
        path = os.path.join(path, 'groups.json') if os.path.isdir(path) else path
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(tokens), f, ensure_ascii=False)

    # ---- queries
    @property
    def vocab_size(self) -> int:
        return len(self.token_group)

    @property
    def num_groups(self) -> int:
        return max(self.token_group, default=-1) + 1

    @cached_property
    def _members(self) -> tuple[tuple[int, ...], ...]:
        members: list[dict[int, int]] = [{} for _ in range(self.num_groups)]
        for t, (g, m) in enumerate(zip(self.token_group, self.token_member)):
            members[g][m] = t
        return tuple(tuple(group[m] for m in sorted(group)) for group in members)

    def groups(self) -> list[list[int]]:
        """Token ids of every group, in member order"""
        return [list(members) for members in self._members]

    def members(self, token: int) -> list[int]:
        """Token ids of the token's group, in member order (the token included)"""
        return list(self._members[self.token_group[token]])

    def group_sizes(self) -> dict[int, int]:
        """Group size -> number of groups of that size"""
        return dict(sorted(Counter(map(len, self._members)).items()))

    def pair(self, token: int) -> tuple[int, int]:
        return self.token_group[token], self.token_member[token]

    def token(self, group: int, member: int) -> int:
        return self._members[group][member]


def build_groups(links: LinkCounts, config: GroupingConfig = GroupingConfig(), reserved: Iterable[int] = ()) -> Grouping:
    """Greedy grouping of aligned subword pairs, strongest Dice score first.

    For each pair (a, b) linked at least min_link_count times with Dice >= min_dice, in decreasing Dice order:
    neither grouped -> a new group [a, b]; exactly one grouped -> the other joins that group unless it is full;
    both grouped -> nothing (groups are never merged, so loosely related words can't chain into large groups).
    Reserved tokens (special tokens) are never grouped; every token left over becomes a group of its own."""
    reserved = set(reserved)
    scored = sorted(((links.dice(a, b), a, b) for (a, b), count in links.links.items()
                     if count >= config.min_link_count and a not in reserved and b not in reserved), reverse=True)
    groups: list[list[int]] = []
    group_of: dict[int, int] = {}
    for dice, a, b in scored:
        if dice < config.min_dice:
            break
        ga, gb = group_of.get(a), group_of.get(b)
        if ga is None and gb is None:
            group_of[a] = group_of[b] = len(groups)
            groups.append([a, b])
        elif ga is None or gb is None:
            g, new = (gb, a) if ga is None else (ga, b)
            if len(groups[g]) < config.max_group_size:
                group_of[new] = g
                groups[g].append(new)
    for t in range(links.vocab_size):
        if t not in group_of:
            group_of[t] = len(groups)
            groups.append([t])
    token_member = [0] * links.vocab_size
    for members in groups:
        for m, t in enumerate(members):
            token_member[t] = m
    return Grouping(tuple(group_of[t] for t in range(links.vocab_size)), tuple(token_member), config.max_group_size)
