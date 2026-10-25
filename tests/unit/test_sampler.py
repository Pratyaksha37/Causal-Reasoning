"""Unit tests for document-level candidate sampler and holdout splits."""

from src.annotation.sampler import partition_documents


def test_partition_documents_no_leakage() -> None:
    doc_ids = [f"DOC_{i:03d}" for i in range(100)]
    splits = partition_documents(
        doc_ids, train_ratio=0.60, dev_ratio=0.20, test_ratio=0.20, seed=42
    )

    train_set = set(splits["train"])
    dev_set = set(splits["dev"])
    test_set = set(splits["test"])

    # Strict partition: disjoint sets
    assert len(train_set & dev_set) == 0
    assert len(train_set & test_set) == 0
    assert len(dev_set & test_set) == 0
    assert len(train_set | dev_set | test_set) == 100

    assert len(train_set) == 60
    assert len(dev_set) == 20
    assert len(test_set) == 20


def test_partition_documents_seed_reproducible() -> None:
    doc_ids = [f"DOC_{i:03d}" for i in range(50)]
    s1 = partition_documents(doc_ids, seed=999)
    s2 = partition_documents(doc_ids, seed=999)
    assert s1 == s2
