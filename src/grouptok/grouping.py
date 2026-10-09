"""Groupings: a lossless (group, member) code for every token of a vocabulary, and how translation links become one."""
from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .align import LinkCounts

PathLike = str | os.PathLike


@dataclass(frozen=True)
class GroupingConfig:
    """Settings for turning alignment links into groups.

    Attributes:
        max_group_size: The most members a group can have (a model's member head has this many outputs); at least 2.
        min_link_count: How often a subword pair must be aligned to be grouped; at least 1.
        min_dice: The lowest Dice score, `2 * links / (occurrences(a) + occurrences(b))`, for a subword pair to be
            grouped; between 0 and 1.

    Raises:
        ValueError: If a setting is out of range.

    Examples:
        >>> from grouptok import GroupingConfig, TokenizerConfig
        >>> config = TokenizerConfig(grouping=GroupingConfig(max_group_size=4, min_dice=0.2))
        >>> GroupingConfig(max_group_size=1)
        Traceback (most recent call last):
        ...
        ValueError: max_group_size must be at least 2
    """
    max_group_size: int = 8
    min_link_count: int = 5
    min_dice: float = 0.1

    def __post_init__(self) -> None:
        if self.max_group_size < 2:
            raise ValueError('max_group_size must be at least 2')
        if self.min_link_count < 1:
            raise ValueError('min_link_count must be at least 1')
        if not 0 <= self.min_dice <= 1:
            raise ValueError('min_dice must be between 0 and 1')


@dataclass(frozen=True)
class Grouping:
    """A one-to-one map between token ids and `(group, member)` pairs.

    Token `t` is the pair `(token_group[t], token_member[t])`. Members of a group are numbered 0, 1, ... in the order
    they joined it, and tokens without a translation are groups of their own (member 0).

    Attributes:
        token_group: The group id of each token; group ids are 0 to `num_groups - 1`.
        token_member: The member id of each token within its group.
        max_group_size: The bound on member ids (the size of a model's member table); can exceed the largest group.

    Raises:
        ValueError: If the two tuples differ in length, the group ids are not 0 to `num_groups - 1`, a member id is
            not below `max_group_size`, or two tokens have the same pair.

    Examples:
        >>> from grouptok import Grouping
        >>> g = Grouping.from_groups([[0], [3, 1], [2]])   # tokens 3 and 1 translate each other
        >>> g.pair(1)
        (1, 1)
        >>> g.token(1, 1)
        1
        >>> g.members(3)
        [3, 1]
    """
    token_group: tuple[int, ...]
    token_member: tuple[int, ...]
    max_group_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'token_group', tuple(self.token_group))
        object.__setattr__(self, 'token_member', tuple(self.token_member))
        if len(self.token_group) != len(self.token_member):
            raise ValueError('token_group and token_member must have the same length')
        if set(self.token_group) != set(range(self.num_groups)):
            raise ValueError('group ids must be 0, 1, ..., number of groups - 1')
        if any(not 0 <= m < self.max_group_size for m in self.token_member):
            raise ValueError(f'member ids must be between 0 and max_group_size - 1 = {self.max_group_size - 1}')
        if len(set(zip(self.token_group, self.token_member))) != len(self.token_group):
            raise ValueError('two tokens have the same (group, member) pair')

    # ---- construction
    @classmethod
    def flat(cls, vocab_size: int) -> Grouping:
        """Make a grouping where every token is a group of its own.

        Args:
            vocab_size: The number of tokens.

        Returns:
            A grouping of `vocab_size` one-member groups, with `max_group_size` 1.

        Examples:
            >>> Grouping.flat(3).groups()
            [[0], [1], [2]]
        """
        return cls(tuple(range(vocab_size)), (0,) * vocab_size, 1)

    @classmethod
    def from_groups(cls, groups: Sequence[Sequence[int]], max_group_size: int | None = None) -> Grouping:
        """Build a grouping from lists of token ids.

        Args:
            groups: The token ids of each group, in member order. Together they must cover the tokens 0 to
                vocabulary size - 1 exactly once.
            max_group_size: The bound on member ids; defaults to the size of the largest group.

        Returns:
            The grouping, with group ids in the order of `groups`.

        Raises:
            ValueError: If a token id is out of range or in more than one group.

        Examples:
            >>> g = Grouping.from_groups([[0], [3, 1], [2]], max_group_size=8)
            >>> g.token_group, g.token_member
            ((0, 1, 2, 1), (0, 1, 0, 0))
        """
        vocab_size = sum(map(len, groups))
        token_group, token_member = [-1] * vocab_size, [0] * vocab_size
        for g, members in enumerate(groups):
            for m, t in enumerate(members):
                if not 0 <= t < vocab_size or token_group[t] != -1:
                    raise ValueError(f'token {t} is out of range or in more than one group')
                token_group[t], token_member[t] = g, m
        return cls(tuple(token_group), tuple(token_member), max_group_size or max(map(len, groups), default=1))

    @classmethod
    def load(cls, path: PathLike) -> Grouping:
        """Read a grouping from `groups.json`.

        Only `token_group`, `token_member` and `max_group_size` are read; other keys are ignored.

        Args:
            path: The file, or a directory containing `groups.json`.

        Returns:
            The grouping.

        Raises:
            ValueError: If the file holds an invalid grouping.

        Examples:
            >>> g = Grouping.load("my-tokenizer")   # same as Grouping.load("my-tokenizer/groups.json")
        """
        path = os.path.join(path, 'groups.json') if os.path.isdir(path) else path
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return cls(tuple(data['token_group']), tuple(data['token_member']), int(data['max_group_size']))

    def save(self, path: PathLike, tokens: Sequence[str] | None = None) -> None:
        """Write the grouping to `groups.json`.

        Args:
            path: The file, or an existing directory to write `groups.json` into.
            tokens: The vocabulary's tokens, indexed by id. If given, the file also gets a `groups` map of every group
                with several members, for people reading it.

        Examples:
            >>> g = Grouping.from_groups([[0], [3, 1], [2]])
            >>> g.save("groups.json", tokens=["<pad>", "Ġparc", "Ġthe", "Ġpark"])
            >>> Grouping.load("groups.json") == g
            True
        """
        path = os.path.join(path, 'groups.json') if os.path.isdir(path) else path
        data: dict[str, Any] = {'max_group_size': self.max_group_size, 'token_group': list(self.token_group),
                                'token_member': list(self.token_member)}
        if tokens is not None:
            data['groups'] = {g: [tokens[t] for t in members] for g, members in enumerate(self.groups()) if len(members) > 1}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    # ---- queries
    @property
    def vocab_size(self) -> int:
        """The number of tokens.

        Examples:
            >>> Grouping.from_groups([[0], [3, 1], [2]]).vocab_size
            4
        """
        return len(self.token_group)

    @property
    def num_groups(self) -> int:
        """The number of groups.

        Examples:
            >>> Grouping.from_groups([[0], [3, 1], [2]]).num_groups
            3
        """
        return max(self.token_group, default=-1) + 1

    @cached_property
    def _members(self) -> tuple[tuple[int, ...], ...]:
        members: list[dict[int, int]] = [{} for _ in range(self.num_groups)]
        for t, (g, m) in enumerate(zip(self.token_group, self.token_member)):
            members[g][m] = t
        return tuple(tuple(group[m] for m in sorted(group)) for group in members)

    def groups(self) -> list[list[int]]:
        """List the token ids of every group.

        Returns:
            The token ids of each group in member order, indexed by group id.

        Examples:
            >>> Grouping.from_groups([[0], [3, 1], [2]]).groups()
            [[0], [3, 1], [2]]
        """
        return [list(members) for members in self._members]

    def members(self, token: int) -> list[int]:
        """List the token ids of a token's group.

        Args:
            token: A token id.

        Returns:
            The token ids of its group in member order, the token included.

        Examples:
            >>> Grouping.from_groups([[0], [3, 1], [2]]).members(1)
            [3, 1]
        """
        return list(self._members[self.token_group[token]])

    def group_sizes(self) -> dict[int, int]:
        """Count the groups of each size.

        Returns:
            The number of groups of each group size, keyed by size in increasing order.

        Examples:
            >>> Grouping.from_groups([[0], [3, 1], [2]]).group_sizes()
            {1: 2, 2: 1}
        """
        return dict(sorted(Counter(map(len, self._members)).items()))

    def pair(self, token: int) -> tuple[int, int]:
        """Map a token id to its `(group, member)` pair.

        Args:
            token: A token id.

        Returns:
            The token's `(group, member)` pair.

        Examples:
            >>> Grouping.from_groups([[0], [3, 1], [2]]).pair(3)
            (1, 0)
        """
        return self.token_group[token], self.token_member[token]

    def token(self, group: int, member: int) -> int:
        """Map a `(group, member)` pair back to its token id.

        Args:
            group: A group id.
            member: A member id within that group.

        Returns:
            The token id.

        Raises:
            IndexError: If there is no such group, or the group has no such member.

        Examples:
            >>> Grouping.from_groups([[0], [3, 1], [2]]).token(1, 0)
            3
        """
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
