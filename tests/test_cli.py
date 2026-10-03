"""Reading pair files and the inspect / encode commands."""
from conftest import SAMPLE
from grouptok.cli import main, read_pairs


def test_read_jsonl_pairs():
    pairs = read_pairs(str(SAMPLE))   # the first two text fields: en, fr
    assert len(pairs) == 500
    assert pairs[1] == ('Then Phinehas stood up, and executed judgment, so the plague was stopped.',
                        'Phinées se leva pour intervenir, Et la plaie s`arrêta;')
    assert read_pairs(f'{SAMPLE}:fr,en', limit=2)[1] == pairs[1][::-1]
    assert len(read_pairs(str(SAMPLE), columns=['en', 'fr'], limit=10)) == 10


def test_read_tsv_pairs(tmp_path):
    path = tmp_path / 'pairs.tsv'
    path.write_text('hello\tbonjour\n\nworld\tmonde\nonly one side\t\n', encoding='utf-8')
    assert read_pairs(str(path)) == [('hello', 'bonjour'), ('world', 'monde')]


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
