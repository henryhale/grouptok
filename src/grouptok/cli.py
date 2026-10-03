"""Command line: grouptok train | inspect | encode"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from .align import AlignerConfig
from .grouping import GroupingConfig
from .tokenizer import DEFAULT_SPECIAL_TOKENS, GroupedTokenizer, Pair, TokenizerConfig


def read_pairs(spec: str, columns: Sequence[str] | None = None, limit: int | None = None) -> list[Pair]:
    """Sentence pairs from a local file: .tsv (two tab-separated columns) or .jsonl (one object per line).

    spec is a path, optionally followed by ':first,second' to name the two JSONL fields of this file (default: columns,
    else the first two text fields of the first line)."""
    path, own = spec, None
    head, sep, tail = spec.rpartition(':')
    if sep and ',' in tail and os.sep not in tail:
        path, own = head, tail.split(',')
    columns = own or columns
    pairs: list[Pair] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if limit is not None and len(pairs) >= limit:
                break
            line = line.rstrip('\n')
            if not line.strip():
                continue
            if path.endswith(('.tsv', '.tab')):
                first, _, second = line.partition('\t')
            else:
                row = json.loads(line)
                if columns is None:
                    columns = [k for k, v in row.items() if isinstance(v, str)][:2]
                    print(f'{path}: using columns {columns[0]!r} and {columns[1]!r}', file=sys.stderr)
                first, second = row.get(columns[0]), row.get(columns[1])
            if first and second:
                pairs.append((first, second))
    return pairs


def _train(args: argparse.Namespace) -> None:
    pairs = [pair for spec in args.pairs for pair in read_pairs(spec, args.columns.split(',') if args.columns else None, args.limit)]
    chat_template = open(args.chat_template, encoding='utf-8').read() if args.chat_template else None
    config = TokenizerConfig(
        vocab_size=args.vocab_size, special_tokens=tuple(args.special_tokens.split(',')),
        additional_tokens=tuple(t for t in args.additional_tokens.split(',') if t), chat_template=chat_template,
        grouping=GroupingConfig(max_group_size=args.max_group_size, min_link_count=args.min_link_count, min_dice=args.min_dice),
        aligner=AlignerConfig(model=args.aligner, layer=args.layer, threshold=args.threshold, batch_size=args.batch_size,
                              max_length=args.max_length, device=args.device))
    print(f'{len(pairs)} sentence pairs', file=sys.stderr)
    tok = GroupedTokenizer.train(pairs, config, progress=not args.quiet)
    tok.save_pretrained(args.out)
    print(f'saved to {args.out}')


def _inspect(args: argparse.Namespace) -> None:
    tok = GroupedTokenizer.from_pretrained(args.dir)
    groups = tok.readable_groups(min_size=2)
    print(f'vocabulary: {tok.vocab_size} tokens in {tok.grouping.num_groups} groups ({sum(map(len, groups))} tokens in '
          f'{len(groups)} groups of 2+)')
    print(f'group sizes: {tok.grouping.group_sizes()}')
    for words in args.word or []:
        print(f'{words!r}: {tok.group_of(words)}')
    if not args.word:
        for members in groups[:args.limit]:
            print('  ' + ' | '.join(repr(m) for m in members))


def _encode(args: argparse.Namespace) -> None:
    tok = GroupedTokenizer.from_pretrained(args.dir)
    ids = tok.encode(args.text)
    print(' '.join(f'{tok.id_to_token(i)}({g},{m})' for i, (g, m) in zip(ids, tok.pairs(ids))))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='grouptok', description='Translation-grouped tokenizer')
    sub = parser.add_subparsers(dest='command', required=True)

    train = sub.add_parser('train', help='train a tokenizer on sentence pairs')
    train.add_argument('--pairs', nargs='+', required=True, metavar='FILE[:A,B]',
                       help='.tsv or .jsonl files of sentence pairs; several files (e.g. one per language pair) are pooled')
    train.add_argument('--columns', help='JSONL fields of the two sides, e.g. en,fr (default: the first two text fields)')
    train.add_argument('--limit', type=int, help='at most this many pairs per file')
    train.add_argument('--out', required=True, help='output directory')
    train.add_argument('--vocab-size', type=int, default=8192)
    train.add_argument('--max-group-size', type=int, default=8)
    train.add_argument('--min-link-count', type=int, default=5)
    train.add_argument('--min-dice', type=float, default=0.1)
    train.add_argument('--aligner', default=AlignerConfig.model, help='Hugging Face encoder used to align subwords')
    train.add_argument('--layer', type=int, default=AlignerConfig.layer)
    train.add_argument('--threshold', type=float, default=AlignerConfig.threshold)
    train.add_argument('--batch-size', type=int, default=AlignerConfig.batch_size)
    train.add_argument('--max-length', type=int, default=AlignerConfig.max_length)
    train.add_argument('--device', default=None, help='default: cuda when available')
    train.add_argument('--special-tokens', default=','.join(DEFAULT_SPECIAL_TOKENS), help='pad,bos,eos')
    train.add_argument('--additional-tokens', default='', help='comma-separated reserved tokens, e.g. "<think>,</think>"')
    train.add_argument('--chat-template', help='file with a Jinja chat template')
    train.add_argument('--quiet', action='store_true')
    train.set_defaults(run=_train)

    inspect = sub.add_parser('inspect', help='show the groups of a trained tokenizer')
    inspect.add_argument('dir')
    inspect.add_argument('--word', nargs='+', help='show the groups of these tokens or single-token texts (e.g. " park")')
    inspect.add_argument('--limit', type=int, default=20, help='groups to list')
    inspect.set_defaults(run=_inspect)

    encode = sub.add_parser('encode', help='encode a text and show each token as (group, member)')
    encode.add_argument('dir')
    encode.add_argument('text')
    encode.set_defaults(run=_encode)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == '__main__':
    main()
