"""grouptok: a translation-grouped tokenizer. Subwords that translate each other share a group, and every token is a
lossless (group, member) pair."""
from importlib.metadata import PackageNotFoundError, version

from .align import Aligner, AlignerConfig, HFAligner, LinkCounts, align_pairs
from .grouping import Grouping, GroupingConfig, build_groups
from .tokenizer import DEFAULT_SPECIAL_TOKENS, GroupedTokenizer, TokenizerConfig, train_bpe

try:
    __version__ = version('grouptok')
except PackageNotFoundError:  # running from a source checkout
    __version__ = '0.0.0+unknown'

__all__ = ['GroupedTokenizer', 'TokenizerConfig', 'Grouping', 'GroupingConfig', 'build_groups', 'Aligner', 'AlignerConfig',
           'HFAligner', 'LinkCounts', 'align_pairs', 'train_bpe', 'DEFAULT_SPECIAL_TOKENS', '__version__']
