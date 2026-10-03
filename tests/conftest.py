"""Shared fixtures: the 500-pair English/French sample and an offline stand-in for the neural aligner."""
import hashlib
import json
import re
from pathlib import Path

import pytest

SAMPLE = Path(__file__).parent / 'data' / 'opus-en-fr-500.jsonl'
# translation pairs the fake aligner knows (word pairs that co-occur in at least 23 sentence pairs of the sample)
TRANSLATIONS = [('God', 'Dieu'), ('said', 'dit'), ('LORD', 'Éternel'), ('Israel', 'Israël'), ('king', 'roi'), ('people', 'peuple'),
                ('house', 'maison')]
# the ones whose word-initial forms (' God', ' Dieu', ...) are frequent enough to check: 23-50 sentence pairs with both;
# the French text mostly writes l`Éternel and d`Israël, so ' Éternel' and ' Israël' are rare
CHECKED = [('God', 'Dieu'), ('said', 'dit'), ('king', 'roi'), ('people', 'peuple'), ('house', 'maison')]


class FakeAligner:
    """Offline stand-in for a neural aligner: its word pieces are words and punctuation marks; the words of each pair in
    TRANSLATIONS share a vector, and every other word gets a fixed pseudo-random one"""
    dim = 32

    def __init__(self):
        self.concepts = {word.lower(): k for k, pair in enumerate(TRANSLATIONS) for word in pair}

    def vector(self, word):
        import torch
        concept = self.concepts.get(word.lower())
        if concept is not None:
            v = torch.zeros(self.dim)
            v[concept] = 8.0
            return v
        seed = int.from_bytes(hashlib.md5(word.encode()).digest()[:8], 'little')
        return 0.3 * torch.randn(self.dim, generator=torch.Generator().manual_seed(seed))

    def encode(self, texts):
        import torch
        pieces = [[(m.start(), m.end(), m.group()) for m in re.finditer(r'\w+|[^\w\s]', text)] for text in texts]
        length = max(map(len, pieces)) + 2  # room for the encoder's own start and end tokens, which have empty spans
        hidden = torch.zeros(len(texts), length, self.dim)
        spans = torch.zeros(len(texts), length, 2, dtype=torch.long)
        for b, sentence in enumerate(pieces):
            for i, (start, end, word) in enumerate(sentence, start=1):
                hidden[b, i] = self.vector(word)
                spans[b, i] = torch.tensor([start, end])
        return hidden, spans


@pytest.fixture(scope='session')
def pairs():
    with open(SAMPLE, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    return [(row['en'], row['fr']) for row in rows]


@pytest.fixture(scope='session')
def trained(pairs):
    """A tokenizer trained on the sample with the fake aligner"""
    pytest.importorskip('torch')
    from grouptok import AlignerConfig, GroupedTokenizer, TokenizerConfig
    config = TokenizerConfig(vocab_size=1500, additional_tokens=('<think>', '</think>'), chat_template='{{ messages }}',
                             aligner=AlignerConfig(device='cpu'))
    return GroupedTokenizer.train(pairs, config, aligner=FakeAligner(), progress=False)
