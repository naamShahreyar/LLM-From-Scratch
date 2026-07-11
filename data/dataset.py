import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import BPETokenizer


class TextDataset(Dataset):

    def __init__(self, token_ids, max_length, stride):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride
        self.num_samples = (len(token_ids) - max_length) // stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start : start + self.max_length]
        y = self.data[start + 1 : start + self.max_length + 1]
        return x, y


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


def create_dataloader_from_ids(
    token_ids,
    max_length,
    stride,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    drop_last=True,
):
    dataset = TextDataset(token_ids, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
