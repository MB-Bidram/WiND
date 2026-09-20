"""ByteTokenizer and TextCollator for basic byte-level tokenization."""

import torch
from torch.utils.data import default_collate


def _build_byte_tables():
    """Build a byte -> token_id and token_id -> byte mapping."""
    bytes_list = bytes(range(256))
    id_to_token = {i: bytes([b]) for i, b in enumerate(bytes_list)}
    token_to_id = {v: k for k, v in id_to_token.items()}
    return id_to_token, token_to_id


class ByteTokenizer:
    """Simple byte-level tokenizer: maps each byte to a unique ID."""

    def __init__(self, pad_token_id: int = 0, bos_token_id: int = 1, eos_token_id: int = 2):
        if len({pad_token_id, bos_token_id, eos_token_id}) != 3:
            raise ValueError("pad, bos, eos IDs must be distinct")
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.byte_stoi = {}
        self.byte_itos = {}
        special = {b"\x00": pad_token_id, b"\x01": bos_token_id, b"\x02": eos_token_id}
        next_id = 3
        for b in range(256):
            byte_val = bytes([b])
            if byte_val in special:
                self.byte_stoi[byte_val] = special[byte_val]
            else:
                self.byte_stoi[byte_val] = next_id
                next_id += 1
        self.vocab_size = next_id
        self.byte_itos = {v: k for k, v in self.byte_stoi.items()}

    def encode(self, text: str) -> torch.Tensor:
        if isinstance(text, str):
            text = text.encode("utf-8")
        return torch.tensor([self.byte_stoi.get(bytes([b]), self.pad_token_id) for b in text], dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return b"".join(self.byte_itos.get(i, b"") for i in ids).decode("utf-8", errors="replace")


class TextCollator:
    """Collate variable-length token sequences into padded batches."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        if isinstance(batch[0], dict):
            return {k: self._pad([b[k] for b in batch], self.pad_token_id if k == "input_ids" else 0)
                    for k in batch[0]}
        return self._pad(batch, self.pad_token_id)

    def _pad(self, items, pad_value):
        max_len = max(x.size(-1) if isinstance(x, torch.Tensor) else len(x) for x in items)
        padded = []
        for item in items:
            if isinstance(item, torch.Tensor):
                pad_len = max_len - item.size(-1)
                if pad_len > 0:
                    item = torch.cat([item, torch.full((pad_len,), pad_value, dtype=item.dtype)])
                padded.append(item)
            else:
                padded.append(torch.tensor(item, dtype=torch.long))
        return torch.stack(padded)


