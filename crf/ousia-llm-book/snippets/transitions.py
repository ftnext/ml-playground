import torch


def create_transitions(
    label2id: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    >>> label2id = {"O": 0, "B-人名": 1, "I-人名": 2, "B-組織名": 3, "I-組織名": 4}
    >>> st, t, et = create_transitions(label2id)
    >>> assert torch.allclose(st, torch.tensor([0, 0, -100, 0, -100], dtype=torch.float))
    >>> assert torch.allclose(et, torch.zeros(5))
    >>> expected = [[0, 0, -100, 0, -100], [0, 0, 0, 0, -100], [0, 0, 0, 0, -100], [0, 0, -100, 0, 0], [0, 0, -100, 0, 0]]
    >>> assert torch.allclose(t, torch.tensor(expected, dtype=torch.float))
    """
    b_ids = [v for k, v in label2id.items() if k.startswith("B")]
    i_ids = [v for k, v in label2id.items() if k.startswith("I")]
    o_id = label2id["O"]

    # 開始からはBとOに遷移可能。Iには遷移不可能
    start_transitions = torch.full([len(label2id)], -100.0)
    start_transitions[b_ids] = 0
    start_transitions[o_id] = 0

    between_labels_transitions = torch.full(
        [len(label2id), len(label2id)], -100.0
    )
    # すべてのラベルからBやOに遷移可能
    between_labels_transitions[:, b_ids] = 0
    between_labels_transitions[:, o_id] = 0
    # Bから同じタイプのIへ、Iから同じタイプのIへ遷移可能
    between_labels_transitions[b_ids, i_ids] = 0
    between_labels_transitions[i_ids, i_ids] = 0

    # すべてのラベルから終了に遷移可能
    end_transitions = torch.zeros(len(label2id))
    return start_transitions, between_labels_transitions, end_transitions
