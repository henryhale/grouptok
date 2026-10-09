"""Subword alignment: which BPE subwords translate each other across a parallel corpus.

An aligner turns sentences into contextual vectors of its own word pieces, with their character spans. Each BPE subword
gets the average of the vectors of the word pieces that overlap it, and two subwords of a sentence pair are linked when
the softmax of their similarity exceeds a threshold in both directions (awesome-align's extraction rule). torch and
transformers are needed here (pip install grouptok[train]); they are imported only when alignment runs."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import tokenizers
    from torch import Tensor


@dataclass(frozen=True)
class AlignerConfig:
    """The neural aligner and the link extraction"""
    model: str = 'aneuraz/awesome-align-with-co'   # any Hugging Face encoder with a fast tokenizer (mBERT, XLM-R, awesome-align, ...)
    layer: int = 8                                  # hidden layer whose vectors are compared
    threshold: float = 1e-3                         # bidirectional softmax threshold for a link
    batch_size: int = 64                            # sentence pairs per batch
    max_length: int = 128                           # aligner input length in word pieces; longer sentences are truncated
    device: str | None = None                       # None = cuda when available
    fp16: bool = True                               # run the encoder in half precision on cuda (the similarity math stays fp32)


class Aligner(Protocol):
    """Anything that maps sentences to word-piece vectors and their character spans"""

    def encode(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """-> (vectors [B, L, H], spans [B, L, 2]) on one device. Spans are character offsets into each text;
        (0, 0) marks positions that are not word pieces (special tokens, padding)"""
        ...


class HFAligner:
    """Word-piece vectors from a layer of a Hugging Face encoder"""

    def __init__(self, config: AlignerConfig = AlignerConfig()) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.config = config
        self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModel.from_pretrained(config.model).to(self.device).eval()

    def encode(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        import torch
        enc = self.tokenizer(texts, return_offsets_mapping=True, padding=True, truncation=True, max_length=self.config.max_length,
                             return_tensors='pt')
        spans = enc.pop('offset_mapping').to(self.device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=self.config.fp16 and str(self.device).startswith('cuda')):
            hidden = self.model(**enc.to(self.device), output_hidden_states=True).hidden_states[self.config.layer]
        return hidden.float(), spans


@dataclass(frozen=True)
class LinkCounts:
    """Alignment statistics of a parallel corpus over a BPE vocabulary"""
    vocab_size: int
    links: dict[tuple[int, int], int]   # (first-side subword, second-side subword) -> times linked; identical subwords excluded
    occurrences: tuple[int, ...]        # subword -> occurrences on either side (positions the aligner covered)

    def dice(self, a: int, b: int) -> float:
        """Dice coefficient of a link: 2 * links / (occurrences of a + occurrences of b)"""
        return 2 * self.links.get((a, b), 0) / (self.occurrences[a] + self.occurrences[b])


def subword_vectors(texts: list[str], bpe: tokenizers.Tokenizer, aligner: Aligner) -> tuple[Tensor, Tensor, Tensor]:
    """One side of a batch -> padded BPE ids [B, N], covered mask [B, N] and BPE vectors [B, N, H]. Each subword gets the
    average of the word pieces whose character spans overlap it; padding and subwords without a word piece (truncated or
    whitespace-only) are not covered"""
    import torch
    hidden, spans = aligner.encode(texts)
    device = hidden.device
    encodings = bpe.encode_batch(texts)
    n = max(len(e.ids) for e in encodings)
    ids = torch.tensor([e.ids + [0] * (n - len(e.ids)) for e in encodings], device=device)
    bpe_spans = torch.tensor([e.offsets + [(0, 0)] * (n - len(e.ids)) for e in encodings], device=device).view(len(texts), n, 2)
    piece_valid = spans[..., 1] > spans[..., 0]  # special tokens and padding have empty spans
    overlap = ((spans[:, None, :, 0] < bpe_spans[:, :, None, 1]) & (bpe_spans[:, :, None, 0] < spans[:, None, :, 1])
               & piece_valid[:, None, :]).float()  # [B, N, L]; padded BPE spans (0, 0) overlap nothing
    count = overlap.sum(-1)
    return ids, count > 0, (overlap @ hidden) / count.clamp(min=1).unsqueeze(-1)


def align_pairs(pairs: Sequence[tuple[str, str]], bpe: tokenizers.Tokenizer, config: AlignerConfig = AlignerConfig(), *,
                aligner: Aligner | None = None, progress: bool = True) -> LinkCounts:
    """Count subword links over sentence pairs, a batch at a time on the aligner's device.

    A subword pair (i, j) of a sentence pair is linked when softmax_j(S[i]) and softmax_i(S[:, j]) both exceed
    config.threshold, where S is the similarity of their vectors; each sentence's softmax spans only its own subwords.
    The aligner defaults to HFAligner(config)."""
    import torch
    aligner = aligner or HFAligner(config)
    vocab_size = bpe.get_vocab_size()
    links: Counter[tuple[int, int]] = Counter()
    occurrences = torch.zeros(vocab_size, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(pairs), config.batch_size):
            first, second = (list(side) for side in zip(*pairs[start:start + config.batch_size]))
            first_ids, first_cov, first_vec = subword_vectors(first, bpe, aligner)
            second_ids, second_cov, second_vec = subword_vectors(second, bpe, aligner)
            seen = torch.bincount(first_ids[first_cov], minlength=vocab_size) + torch.bincount(second_ids[second_cov], minlength=vocab_size)
            occurrences += seen.cpu()
            # uncovered positions are -inf, so each sentence's softmax only spans its own subwords (all-masked rows -> 0)
            sim = (first_vec @ second_vec.transpose(1, 2)).masked_fill(~(first_cov[:, :, None] & second_cov[:, None, :]), float('-inf'))
            linked = (sim.softmax(-1).nan_to_num() > config.threshold) & (sim.softmax(1).nan_to_num() > config.threshold)
            b, i, j = linked.nonzero(as_tuple=True)
            a_tok, b_tok = first_ids[b, i], second_ids[b, j]
            keep = a_tok != b_tok  # identical subwords on both sides are already one token
            links.update(zip(a_tok[keep].tolist(), b_tok[keep].tolist()))
            if progress and start // config.batch_size % 100 == 0:
                print(f'aligned {start + len(first)}/{len(pairs)} pairs, {len(links)} distinct links')
    return LinkCounts(vocab_size, dict(links), tuple(occurrences.tolist()))
