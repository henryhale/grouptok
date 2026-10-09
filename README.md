<div align=center>

# grouptok

A translation-grouped tokenizer: subwords that translate each other share a group

[![PyPI](https://img.shields.io/pypi/v/grouptok)](https://pypi.org/project/grouptok/)
[![Python](https://img.shields.io/pypi/pyversions/grouptok)](https://pypi.org/project/grouptok/)
[![Tests](https://github.com/henryhale/grouptok/actions/workflows/tests.yml/badge.svg)](https://github.com/henryhale/grouptok/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/henryhale/grouptok)](https://github.com/henryhale/grouptok/blob/master/LICENSE)

</div>

**grouptok** trains a byte-level BPE tokenizer on sentence pairs. It then aligns the subwords of each pair with a neural
word aligner and puts subwords that are often aligned into one **group**. Each token becomes a `(group, member)` pair:

```
 " house" -> (412, 0)      " maison" -> (412, 1)      " Haus" -> (412, 2)
 " king"  -> (977, 0)      " roi"    -> (977, 1)
```

A language model can then predict the group first and the member second, so translations share most of their
parameters. The tokenizer itself is an ordinary BPE tokenizer (it saves a standard `tokenizer.json`, so it also loads in
`transformers`), plus a `groups.json` that holds the grouping.

## Install

```bash
pip install grouptok            # load and use a trained tokenizer (needs only `tokenizers`)
pip install "grouptok[train]"   # train one (adds torch and transformers for the aligner)
```

## Training a tokenizer

Training needs the `train` extra. It runs on a GPU when one is available.

```python
from grouptok import GroupedTokenizer, TokenizerConfig

pairs = [("The king said unto the people", "Le roi dit au peuple"), ...]   # (one language, the other)
tok = GroupedTokenizer.train(pairs, TokenizerConfig(vocab_size=8192))
tok.save_pretrained("my-tokenizer")      # tokenizer.json, tokenizer_config.json, groups.json
```

## Using a trained tokenizer

Load a tokenizer saved in a local folder:

```python
from grouptok import GroupedTokenizer

tok = GroupedTokenizer.from_pretrained("my-tokenizer")
```

or one published on the Hugging Face Hub (see [Publishing a tokenizer](#publishing-a-tokenizer)). 
`from_pretrained` reads local folders only, so download the repository first; `huggingface_hub` comes with `tokenizers`:

```python
from huggingface_hub import snapshot_download

tok = GroupedTokenizer.from_pretrained(snapshot_download("your-name/my-tokenizer"))
```

Then:

```python
ids = tok.encode("The king", bos=True, eos=True)
tok.pairs(ids)                           # [(group, member), ...] for each token
tok.decode(ids)                          # 'The king'
tok.group_of(" king")                    # [' king', ' roi']
tok.readable_groups()[:5]                # groups with at least two members, as token strings
```

The grouping is a `Grouping`, a one-to-one map between token ids and `(group, member)` pairs, which a model needs for
its embedding tables and heads:

```python
g = tok.grouping
g.token_group, g.token_member            # per-token group and member ids (tuples of length vocab_size)
g.num_groups, g.max_group_size           # sizes of the group and member tables
g.token(group, member)                   # back to the token id
```

## Publishing a trained tokenizer

Save the trained tokenizer and upload the folder to a repository on the Hugging Face Hub with the
[`hf` command](https://huggingface.co/docs/huggingface_hub/guides/cli). `hf upload` creates the repository if it doesn't
exist yet.

```bash
hf auth login                                        # once, with a token that has write access
hf upload your-name/my-tokenizer my-tokenizer        # repository id, then the folder save_pretrained wrote
```

If `hf` isn't found, update `huggingface_hub`: `pip install -U huggingface_hub`.

Anyone can then download it and load it with grouptok:

```python
from huggingface_hub import snapshot_download
from grouptok import GroupedTokenizer

tok = GroupedTokenizer.from_pretrained(snapshot_download("your-name/my-tokenizer"))
```

The repository holds a standard `tokenizer.json` and `tokenizer_config.json`, so
`transformers.AutoTokenizer.from_pretrained("your-name/my-tokenizer")` also works. 
It loads the BPE tokenizer only and ignores `groups.json`.

## More than two languages

Pairs don't have to share a language pair: pool `en–fr`, `en–de` and `fr–de` pairs and train once. 
A token joins the group of whatever it is most often aligned with, so a group can hold `" house"`, `" maison"` and `" Haus"`. 

The group size cap(`max_group_size`, default 8) bounds how many members a group can get.

```python
from grouptok import GroupedTokenizer, GroupingConfig, TokenizerConfig

pairs = en_fr_pairs + en_de_pairs + fr_de_pairs
tok = GroupedTokenizer.train(pairs, TokenizerConfig(vocab_size=32000, grouping=GroupingConfig(max_group_size=8)))
```

## Configuration

```python
from grouptok import AlignerConfig, GroupingConfig, TokenizerConfig

TokenizerConfig(
    vocab_size=8192,
    special_tokens=("<|endoftext|>", "<|im_start|>", "<|im_end|>"),   # pad, bos, eos -> ids 0, 1, 2
    additional_tokens=("<think>", "</think>"),                        # more reserved tokens, never grouped
    chat_template=None,                                               # Jinja template saved to tokenizer_config.json
    grouping=GroupingConfig(max_group_size=8, min_link_count=5, min_dice=0.1),
    aligner=AlignerConfig(model="aneuraz/awesome-align-with-co", layer=8, threshold=1e-3,
                          batch_size=64, max_length=128, device=None, fp16=True),
)
```

Any Hugging Face encoder with a fast tokenizer can be the aligner ([mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased),
[XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base),
[awesome-align](https://huggingface.co/aneuraz/awesome-align-with-co), ...). You can also pass
your own object to `GroupedTokenizer.train(..., aligner=...)`. It needs an `encode(texts)` method that returns the
vectors `[batch, length, dim]` and the character spans `[batch, length, 2]` of its word pieces, with empty spans for
special and padding positions.

## Command line

```bash
grouptok train --pairs europarl.jsonl:en,fr opus.tsv --limit 200000 --out my-tokenizer --vocab-size 8192
grouptok inspect my-tokenizer --word " king" " house"
grouptok encode my-tokenizer "The king said"
```

`--pairs` takes JSONL files (the two sides are the `--columns`, or the first two text fields) or TSV files (one pair per
line, split at the first tab). Write `FILE:A,B` to choose the fields of one file. `grouptok train --help` lists every
option.

## How it works

1. **BPE.** A byte-level BPE is trained on both sides of every pair. Reserved tokens come first.
2. **Alignment.** Each pair goes through the aligner. A BPE subword gets the mean vector of the aligner's word pieces
   that overlap it in characters. Two subwords are linked when each one's softmax over the other side's similarities
   exceeds `threshold` (awesome-align's rule).
3. **Scores.** The Dice score of two subwords is `2 * links / (occurrences(a) + occurrences(b))`. Identical subwords on
   the two sides (names, numbers) are not linked.
4. **Grouping.** Links are visited by descending Dice, skipping pairs below `min_link_count` or `min_dice`:
   - if neither subword is grouped, they start a new group;
   - if one is, the other joins its group unless the group already has `max_group_size` members;
   - two existing groups are never merged.

   *Every remaining token, including the reserved tokens, is a group of its own.*

## Files

`save_pretrained` writes:
- `tokenizer.json`: a standard `tokenizers` file;
- `tokenizer_config.json`: for `transformers.AutoTokenizer`;
- `groups.json`:

```json
{"max_group_size": 8, "token_group": [0, 1, 2, 3, ...], "token_member": [0, 0, 0, 0, ...],
 "groups": {"412": ["Ġhouse", "Ġmaison", "ĠHaus"], ...}}
```

`groups` is only there for people reading the file. It lists the groups with several members as raw BPE tokens, where
`Ġ` marks a leading space. 
`Grouping.load` reads `token_group` and `token_member`, and ignores any other keys.

## Development

Clone the repo
```bash
git clone https://github.com/henryhale/grouptok.git
cd grouptok
```

Install dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU torch is enough for the tests
pip install -e ".[train,test]"
pytest
```

The tests run offline. They train on a 500-pair English–French sample in `tests/data` (kept in the repository, not in
the package) with a stand-in aligner instead of a downloaded model.

## License

Released under the Apache-2.0 license. See [LICENSE](https://github.com/henryhale/grouptok/blob/master/LICENSE) for details.

&copy; 2026-present [Henry Hale](https://github.com/henryhale)
