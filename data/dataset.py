import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import BPETokenizer


class TextDataset(Dataset):

    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text,
    tokenizer,
    max_length,
    stride,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    drop_last=True,
):
    token_ids = tokenizer.encode(text)
    dataset = TextDataset(token_ids, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
