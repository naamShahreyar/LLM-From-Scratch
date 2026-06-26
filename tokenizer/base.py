from abc import ABC, abstractmethod


class Tokenizer(ABC):

    @abstractmethod
    def train(self, text: str, vocab_size: int) -> None: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...
