"""The aligner's windows over long sentences, and batching by length."""
import re

import pytest

from conftest import FakeAligner
from grouptok import AlignerConfig


def tiny_encoder(path):
    """A tiny random BERT whose tokenizer maps every word to [UNK], saved where HFAligner can load it offline"""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token='[PAD]', unk_token='[UNK]', cls_token='[CLS]',
                            sep_token='[SEP]').save_pretrained(path)
    BertModel(BertConfig(vocab_size=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=16)).save_pretrained(path)


def test_long_sentences_are_encoded_in_windows(tmp_path):
    """Every word piece of a sentence longer than max_length comes back, none is cut"""
    pytest.importorskip('transformers')
    from grouptok import HFAligner
    tiny_encoder(tmp_path)
    aligner = HFAligner(AlignerConfig(model=str(tmp_path), layer=1, max_length=8, device='cpu'))
    texts = ['a short one', ' '.join(f'w{i},' for i in range(30)), 'mid sized, text']
    hidden, spans = aligner.encode(texts)
    assert hidden.shape[:2] == spans.shape[:2] and hidden.shape[1] > 8
    for text, row in zip(texts, spans.tolist()):
        assert {(s, e) for s, e in row if e > s} == {m.span() for m in re.finditer(r'\w+|[^\w\s]+', text)}


def test_length_batches(trained, pairs):
    """Batches stay within batch_tokens unless a pair is longer alone, and batching doesn't change the links"""
    from grouptok.align import align_pairs, length_batches
    bpe = trained.tokenizer
    longest = [max(len(bpe.encode(a)), len(bpe.encode(b))) for a, b in pairs]
    budget = max(longest) // 2
    batches = list(length_batches(pairs, bpe, budget))
    assert sorted(i for batch in batches for i in batch) == list(range(len(pairs)))
    assert all(len(batch) * max(longest[i] for i in batch) <= budget or len(batch) == 1 for batch in batches)
    assert any(longest[batch[0]] > budget for batch in batches)   # the longest pairs go alone
    small, large = (align_pairs(pairs, bpe, AlignerConfig(batch_tokens=n, device='cpu'), aligner=FakeAligner(), progress=False)
                    for n in (budget, 10 ** 6))
    assert small == large
