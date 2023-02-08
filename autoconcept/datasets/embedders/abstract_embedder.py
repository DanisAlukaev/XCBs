from abc import ABC, abstractmethod
from typing import List

Token = str
Sequence = str
Embedding = List[float]


class Embedder(ABC):

    @abstractmethod
    def embed_token(self, token: Token) -> Embedding:
        pass

    def batch_embed_token(self, batch_token: List[Token]) -> List[Embedding]:
        return [
            self.embed_token(token)
            for token in batch_token
        ]

    @abstractmethod
    def embed_tokens(self, sequence: List[Token]) -> List[Embedding]:
        pass

    def batch_embed_tokens(self, batch_sequence: List[List[Token]]) -> List[List[Embedding]]:
        return [
            self.embed_tokens(sequence)
            for sequence in batch_sequence
        ]

    @abstractmethod
    def embed_sequence(self, sequence: Sequence) -> Embedding:
        pass

    def batch_embed_sequence(self, batch_sequence: List[Sequence]) -> List[Embedding]:
        return [
            self.embed_sequence(sequence)
            for sequence in batch_sequence
        ]
