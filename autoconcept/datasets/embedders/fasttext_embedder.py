import inspect
import os
from typing import List

import fasttext

from .abstract_embedder import Embedder, Embedding, Sequence, Token


class FastTextEmbedder(Embedder):

    current_dir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    MODEL_PATH = os.path.join(current_dir, "resources/cc.en.300.bin")

    def __init__(self):
        self.model = fasttext.load_model(self.MODEL_PATH)

    def embed_token(self, token: Token) -> Embedding:
        return self.model.get_word_vector(token)

    def embed_tokens(self, tokens: List[Token]) -> List[Embedding]:
        return self.batch_embed_token(tokens)

    def embed_sequence(self, sequence: Sequence) -> Embedding:
        return self.model.get_sentence_vector(sequence)
