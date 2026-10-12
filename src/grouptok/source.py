"""Source: one dataset to train on (a local file or a Hugging Face dataset), with its columns, row count and aligner, so
several datasets can be described once and passed to GroupedTokenizer.train together."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .align import AlignerConfig


@dataclass(frozen=True)
class Source:
    """One dataset of sentence pairs to train on: where it is, which columns to read and how many rows.

    Attributes:
        path: A local `.jsonl` file (one JSON object per line), or a Hugging Face dataset id, streamed (needs
            `pip install "grouptok[datasets]"`).
        columns: The two fields of the pairs, in order; the first one supplies member 0 of the groups it creates. A dot
            reads a nested field (`'translation.fr'`).
        subset: The Hugging Face dataset's configuration (e.g. `'en-fr'`).
        split: The Hugging Face dataset's split.
        rows: Read at most this many rows with text on both sides; `None` reads them all.
        aligner: The aligner for this source's pairs; `None` uses the one `GroupedTokenizer.train` was given.

    Raises:
        ValueError: If `columns` is not two fields, or `rows` is not positive.

    Examples:
        >>> from grouptok import AlignerConfig, GroupedTokenizer, Source, TokenizerConfig
        >>> sources = [
        ...     Source("Helsinki-NLP/opus-100", subset="en-fr", columns=("translation.en", "translation.fr"), rows=200_000),
        ...     Source("pairs/en-sw.jsonl", columns=("en", "sw")),
        ...     Source("Helsinki-NLP/opus-100", subset="en-ha", columns=("translation.en", "translation.ha"),
        ...            aligner=AlignerConfig(model="Davlan/afro-xlmr-large")),
        ... ]
        >>> tok = GroupedTokenizer.train(sources, TokenizerConfig(vocab_size=32000))
    """
    path: str
    columns: tuple[str, str]
    subset: str | None = None
    split: str = 'train'
    rows: int | None = None
    aligner: AlignerConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'path', os.fspath(self.path))
        if self.columns is None or isinstance(self.columns, str) or len(self.columns) != 2:   # a string 'en' would read 'e' and 'n'
            raise ValueError(f'{self.path}: columns must be the two fields of the pairs, e.g. ("en", "fr"), not {self.columns!r}')
        object.__setattr__(self, 'columns', tuple(self.columns))
        if self.rows is not None and self.rows < 1:
            raise ValueError('rows must be positive, or None for all')

    def load(self) -> list[tuple[str, str]]:
        """Read the pairs of the source.

        Returns:
            The first `rows` rows with text on both sides, as `(text, translation)` pairs in column order.

        Raises:
            ValueError: If no row has text on both sides (often a misspelled column name).
            ImportError: If `path` is a Hugging Face dataset and `datasets` is not installed.

        Examples:
            >>> Source("pairs.jsonl", columns=("fr", "en"), rows=2).load()
            [('Le roi dit', 'The king said'), ('Dieu dit', 'God said')]
        """
        out = []
        for record in self._records():
            if self.rows is not None and len(out) >= self.rows:
                break
            pair = tuple(_field(record, c) for c in self.columns)
            if all(isinstance(v, str) and v for v in pair):
                out.append(pair)
        if not out:
            raise ValueError(f'{self.path}: no rows with text in the columns {self.columns}')
        return out

    def _records(self) -> Iterator[dict[str, Any]]:
        """The rows of a local JSONL file, else of the Hugging Face dataset, streamed"""
        if self.path.endswith('.jsonl') or os.path.isfile(self.path):
            with open(self.path, encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f'{self.path}: every line must be a JSON object')
                    yield record
            return
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(f'{self.path} is not a local file; reading Hugging Face datasets needs '
                              '`pip install "grouptok[datasets]"`') from e
        yield from load_dataset(self.path, self.subset, split=self.split, streaming=True)


def _field(record: dict[str, Any], column: str) -> Any:
    """record[column]; unless a field has that exact name, dots step into nested fields ('translation.fr')"""
    if column in record:
        return record[column]
    for key in column.split('.'):
        record = record.get(key) if isinstance(record, dict) else None
    return record
