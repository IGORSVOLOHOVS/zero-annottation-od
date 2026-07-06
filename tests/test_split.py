import pytest

from zeod.dataset.split import train_val_split


def test_split_is_deterministic_for_same_seed():
    items = list(range(100))
    train1, val1 = train_val_split(items, val_split=0.2, seed=42)
    train2, val2 = train_val_split(items, val_split=0.2, seed=42)
    assert train1 == train2
    assert val1 == val2


def test_split_proportions_are_approximately_correct():
    items = list(range(100))
    train, val = train_val_split(items, val_split=0.2, seed=42)
    assert len(val) == 20
    assert len(train) == 80


def test_split_covers_every_item_exactly_once():
    items = list(range(50))
    train, val = train_val_split(items, val_split=0.3, seed=7)
    assert sorted(train + val) == items
    assert set(train).isdisjoint(set(val))


def test_different_seeds_can_produce_different_splits():
    items = list(range(200))
    _, val_a = train_val_split(items, val_split=0.2, seed=1)
    _, val_b = train_val_split(items, val_split=0.2, seed=2)
    assert val_a != val_b


def test_does_not_mutate_input_list():
    items = list(range(20))
    original = list(items)
    train_val_split(items, val_split=0.2, seed=42)
    assert items == original


def test_does_not_leak_into_global_random_state():
    import random

    random.seed(123)
    state_before = random.getstate()
    train_val_split(list(range(30)), val_split=0.2, seed=999)
    assert random.getstate() == state_before


@pytest.mark.parametrize("bad_split", [0, 1, -0.1, 1.5])
def test_invalid_val_split_raises(bad_split):
    with pytest.raises(ValueError):
        train_val_split(list(range(10)), val_split=bad_split, seed=42)
