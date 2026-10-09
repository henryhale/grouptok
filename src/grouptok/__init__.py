"""grouptok: a translation-grouped tokenizer. Subwords that translate each other share a group, and every token is a
lossless (group, member) pair."""
from importlib.metadata import PackageNotFoundError, version

from .align import Aligner, AlignerConfig, HFAligner
from .grouping import Grouping, GroupingConfig
from .source import Source
from .tokenizer import GroupedTokenizer, TokenizerConfig

try:
    __version__ = version('grouptok')
except PackageNotFoundError:  # running from a source checkout
    __version__ = '0.0.0+unknown'

__all__ = ['GroupedTokenizer', 'TokenizerConfig', 'Source', 'Grouping', 'GroupingConfig', 'Aligner', 'AlignerConfig', 'HFAligner', '__version__']
