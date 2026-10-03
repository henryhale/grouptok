"""End-to-end training on the sample with the offline aligner, persistence and encoding."""
import json

import pytest

from conftest import CHECKED
from grouptok import GroupedTokenizer


def single_token(tok, word):
    ids = tok.encode(' ' + word)
    return ids[0] if len(ids) == 1 else None


def test_translations_share_a_group(trained):
    checked = 0
    for en, fr in CHECKED:
        a, b = single_token(trained, en), single_token(trained, fr)
        if a is None or b is None:
            continue
        checked += 1
        assert trained.grouping.token_group[a] == trained.grouping.token_group[b], (en, fr)
    assert checked >= 4, 'too few translation pairs are single tokens to check'


def test_grouping_invariants(trained):
    g = trained.grouping
    assert g.vocab_size == trained.vocab_size <= 1500
    assert max(map(len, g.groups())) <= g.max_group_size == 8
    for token in trained.special_tokens + trained.additional_tokens:  # reserved tokens are never grouped
        assert g.members(trained.token_to_id(token)) == [trained.token_to_id(token)]
    assert (trained.pad_id, trained.bos_id, trained.eos_id) == (0, 1, 2)
    assert any(len(m) > 1 for m in g.groups())


def test_encode_decode(trained, pairs):
    for text in pairs[0]:
        ids = trained.encode(text, bos=True, eos=True)
        assert ids[0] == trained.bos_id and ids[-1] == trained.eos_id
        assert trained.decode(ids) == text
        assert trained.encode_batch([text], bos=True, eos=True) == [ids]
    assert trained.decode(trained.encode('<think> ok')) == '<think> ok'   # additional tokens stay visible
    assert len(trained.pairs(trained.encode(pairs[0][0]))) == len(trained.encode(pairs[0][0]))


def test_group_of(trained):
    en, fr = next((en, fr) for en, fr in CHECKED if single_token(trained, en) is not None and single_token(trained, fr) is not None)
    assert set(trained.group_of(' ' + en)) >= {' ' + en, ' ' + fr}
    assert trained.group_of(single_token(trained, en)) == trained.group_of(' ' + en)
    with pytest.raises(KeyError):
        trained.group_of('a whole sentence of many tokens')
    assert all(len(group) >= 2 for group in trained.readable_groups())


def test_save_and_load(trained, tmp_path):
    trained.save_pretrained(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {'tokenizer.json', 'tokenizer_config.json', 'groups.json'}
    config = json.loads((tmp_path / 'tokenizer_config.json').read_text())
    assert (config['pad_token'], config['bos_token'], config['eos_token']) == trained.special_tokens
    assert config['chat_template'] == '{{ messages }}'
    added = {t['content']: t['special'] for t in json.loads((tmp_path / 'tokenizer.json').read_text())['added_tokens']}
    assert added == {'<|endoftext|>': True, '<|im_start|>': True, '<|im_end|>': True, '<think>': False, '</think>': False}
    loaded = GroupedTokenizer.from_pretrained(tmp_path)
    assert loaded.grouping == trained.grouping
    assert loaded.special_tokens == trained.special_tokens and loaded.additional_tokens == trained.additional_tokens
    assert loaded.chat_template == trained.chat_template
    text = 'And the king said unto the people'
    assert loaded.encode(text) == trained.encode(text)


def test_transformers_tokenizer(trained, tmp_path):
    pytest.importorskip('transformers')
    from transformers import AutoTokenizer
    text = 'And the king said unto the people <think>'
    hf = trained.to_hf()
    assert hf(text, add_special_tokens=False).input_ids == trained.encode(text)
    assert hf.pad_token_id == trained.pad_id and hf.eos_token_id == trained.eos_id
    trained.save_pretrained(tmp_path)
    auto = AutoTokenizer.from_pretrained(tmp_path)
    assert auto(text, add_special_tokens=False).input_ids == trained.encode(text)
    assert auto.decode(trained.encode(text), skip_special_tokens=True) == text   # '<think>' is not a special token


def test_training_needs_pairs():
    with pytest.raises(ValueError):
        GroupedTokenizer.train([('', 'vide'), ('empty', '')])
