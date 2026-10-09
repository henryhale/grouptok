"""Reading sources: local JSONL files, columns and row limits."""
import pytest

from conftest import SAMPLE
from grouptok import Source


def test_read_jsonl(pairs):
    assert Source(SAMPLE, columns=('en', 'fr')).load() == pairs
    assert pairs[1] == ('Then Phinehas stood up, and executed judgment, so the plague was stopped.',
                        'Phinées se leva pour intervenir, Et la plaie s`arrêta;')
    assert Source(SAMPLE, columns=['fr', 'en'], rows=2).load()[1] == pairs[1][::-1]


def test_nested_fields(tmp_path):
    path = tmp_path / 'opus.jsonl'
    path.write_text('{"translation": {"en": "hello", "fr": "bonjour"}}\n\n{"translation": {"en": "world", "fr": ""}}\n',
                    encoding='utf-8')
    assert Source(path, columns=('translation.fr', 'translation.en')).load() == [('bonjour', 'hello')]


def test_invalid_sources(tmp_path):
    for columns in [None, ('en',), ('en', 'fr', 'de'), 'en']:   # the string 'en' is not the fields 'e' and 'n'
        with pytest.raises(ValueError):
            Source(SAMPLE, columns=columns)
    with pytest.raises(ValueError):
        Source(SAMPLE, columns=('en', 'fr'), rows=0)
    with pytest.raises(ValueError):   # a misspelled column: no row has text in it
        Source(SAMPLE, columns=('en', 'french')).load()
    with pytest.raises(FileNotFoundError):
        Source(tmp_path / 'missing.jsonl', columns=('en', 'fr')).load()
