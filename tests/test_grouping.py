"""Grouping invariants and the greedy grouping rules (no torch needed)."""
import json

import pytest

from grouptok import Grouping, GroupingConfig, LinkCounts, build_groups


def test_flat_grouping():
    g = Grouping.flat(5)
    assert g.vocab_size == g.num_groups == 5
    assert g.max_group_size == 1
    assert [g.pair(t) for t in range(5)] == [(t, 0) for t in range(5)]


def test_from_groups_round_trip():
    g = Grouping.from_groups([[0], [3, 1], [2, 4, 5]], max_group_size=4)
    assert g.vocab_size == 6 and g.num_groups == 3 and g.max_group_size == 4
    assert g.pair(1) == (1, 1) and g.token(2, 2) == 5
    assert g.members(4) == [2, 4, 5]
    assert g.groups() == [[0], [3, 1], [2, 4, 5]]
    assert g.group_sizes() == {1: 1, 2: 1, 3: 1}
    assert all(g.token(*g.pair(t)) == t for t in range(g.vocab_size))


@pytest.mark.parametrize('kwargs', [
    dict(token_group=(0, 0), token_member=(0, 0), max_group_size=2),   # two tokens with the same pair
    dict(token_group=(0, 2), token_member=(0, 0), max_group_size=1),   # group ids not contiguous
    dict(token_group=(0, 0), token_member=(0, 2), max_group_size=2),   # member id beyond max_group_size
    dict(token_group=(0,), token_member=(0, 1), max_group_size=2),     # lengths differ
])
def test_invalid_groupings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Grouping(**kwargs)


def test_from_groups_rejects_duplicates():
    with pytest.raises(ValueError):
        Grouping.from_groups([[0, 1], [1]])


def test_save_and_load(tmp_path):
    g = Grouping.from_groups([[0], [1, 2]], max_group_size=8)
    g.save(tmp_path, tokens=['<pad>', 'Ġpark', 'Ġparc'])
    data = json.loads((tmp_path / 'groups.json').read_text())
    assert list(data) == ['max_group_size', 'token_group', 'token_member', 'groups']
    assert data['groups'] == {'1': ['Ġpark', 'Ġparc']}
    assert Grouping.load(tmp_path) == g
    data['merge_members'] = True  # keys added by other tools are ignored
    (tmp_path / 'groups.json').write_text(json.dumps(data))
    assert Grouping.load(tmp_path / 'groups.json') == g


def links(pairs, occurrences, vocab_size=10):
    return LinkCounts(vocab_size, pairs, tuple(occurrences))


def test_build_groups_greedy_rules():
    occurrences = [10] * 10
    counts = links({(1, 2): 9, (1, 3): 8, (4, 3): 7, (5, 6): 6, (5, 2): 10, (7, 8): 4}, occurrences)
    g = build_groups(counts, GroupingConfig(max_group_size=3, min_link_count=5, min_dice=0.1))
    # strongest first: (5, 2) makes [5, 2]; (1, 2) adds 1 -> [5, 2, 1]; (1, 3) can't join a full group -> 3 stays alone;
    # (4, 3) makes [4, 3]; (5, 6) can't join the full group; (7, 8) has too few links
    assert g.members(5) == [5, 2, 1]
    assert g.members(4) == [4, 3]
    assert g.members(6) == [6] and g.members(7) == [7]
    assert g.group_sizes() == {1: 5, 2: 1, 3: 1}


def test_build_groups_never_merges_groups():
    counts = links({(1, 2): 10, (3, 4): 9, (1, 3): 8}, [10] * 10)
    g = build_groups(counts, GroupingConfig(max_group_size=8))
    assert g.members(1) == [1, 2] and g.members(3) == [3, 4]


def test_build_groups_thresholds_and_reserved_tokens():
    counts = links({(1, 2): 10, (3, 4): 6, (0, 5): 10}, [10, 10, 10, 50, 50, 10, 10, 10, 10, 10])
    g = build_groups(counts, GroupingConfig(min_dice=0.2), reserved=[0])
    assert g.members(1) == [1, 2]          # Dice 2 * 10 / 20 = 1.0
    assert g.members(3) == [3]             # Dice 2 * 6 / 100 = 0.12 < 0.2
    assert g.members(0) == [0] and g.members(5) == [5]   # reserved token 0 is never grouped
    assert g.vocab_size == 10 and g.max_group_size == 8


def test_grouping_config_validation():
    with pytest.raises(ValueError):
        GroupingConfig(max_group_size=1)
    with pytest.raises(ValueError):
        GroupingConfig(min_dice=1.5)
