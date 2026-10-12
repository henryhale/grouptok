"""The train command's --pairs specs and JSON config, and the inspect / encode commands."""
import json

import pytest

from conftest import SAMPLE, FakeAligner
from grouptok import AlignerConfig, GroupedTokenizer, Source
from grouptok.cli import main, pairs_source


def test_pairs_spec():
    assert pairs_source(f'{SAMPLE}:fr,en', limit=2) == Source(SAMPLE, columns=('fr', 'en'), rows=2)
    assert pairs_source(str(SAMPLE), ['en', 'fr'], 10) == Source(SAMPLE, columns=('en', 'fr'), rows=10)


def test_config_excludes_the_pairs_options(tmp_path):
    with pytest.raises(SystemExit):
        main(['train', '--config', 'sources.json', '--limit', '10', '--out', str(tmp_path / 'tok')])


def test_train_from_a_config(tmp_path, monkeypatch):
    """A source's aligner overrides only the aligner options it sets; sources without one use them as they are"""
    pytest.importorskip('torch')
    import grouptok.align
    loaded = []

    def fake_hf_aligner(config):
        loaded.append(config)
        return FakeAligner()

    monkeypatch.setattr(grouptok.align, 'HFAligner', fake_hf_aligner)
    config = tmp_path / 'sources.json'
    config.write_text(json.dumps([{'path': str(SAMPLE), 'columns': ['en', 'fr'], 'rows': 256, 'aligner': {'model': 'own', 'max_length': 64}},
                                  {'path': str(SAMPLE), 'columns': ['fr', 'en'], 'rows': 100}]), encoding='utf-8')
    main(['train', '--config', str(config), '--out', str(tmp_path / 'tok'), '--vocab-size', '1000', '--aligner', 'default',
          '--device', 'cpu', '--quiet'])
    assert loaded == [AlignerConfig(model='own', max_length=64, device='cpu'), AlignerConfig(model='default', device='cpu')]
    assert GroupedTokenizer.from_pretrained(tmp_path / 'tok').vocab_size <= 1000


def test_inspect_and_encode(trained, tmp_path, capsys):
    trained.save_pretrained(tmp_path)
    main(['inspect', str(tmp_path), '--limit', '3'])
    out = capsys.readouterr().out
    assert f'vocabulary: {trained.vocab_size} tokens' in out and 'group sizes' in out
    main(['inspect', str(tmp_path), '--word', ' God'])
    assert "' God'" in capsys.readouterr().out
    main(['encode', str(tmp_path), 'And God said'])
    out = capsys.readouterr().out.strip().split()
    assert len(out) == len(trained.encode('And God said')) and all(token.endswith(')') for token in out)
