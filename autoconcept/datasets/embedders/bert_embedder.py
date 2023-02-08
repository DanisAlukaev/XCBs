from typing import List

import numpy as np
import transformers

from .abstract_embedder import Embedder, Embedding, Sequence, Token


class BertEmbedder(Embedder):

    def __init__(self, model_name, device="cpu", batch_size=4):
        self.tokenizer: transformers.BertTokenizerFast = transformers.BertTokenizerFast.from_pretrained(
            model_name)
        self.model: transformers.BertModel = transformers.BertModel.from_pretrained(
            model_name).to(device)
        self.device = device
        self.batch_size = batch_size

    def embed_token(self, token: Token) -> Embedding:
        embedding = self.embed_sequence(token)
        return embedding

    def batch_embed_token(self, batch_token: List[Token]) -> List[Embedding]:
        embeddings = self.batch_embed_sequence(batch_token)
        return embeddings

    def embed_tokens(self, tokens: List[Token]) -> List[Embedding]:
        tokenizer_result = self.tokenizer.encode_plus(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        forward_result = self.model.forward(
            input_ids=tokenizer_result["input_ids"].to(self.device),
            attention_mask=tokenizer_result["attention_mask"].to(self.device),
        )

        tokens_embeddings = []
        prev_offset = [0, 999]
        for embedding, offset_mapping in zip(
            forward_result.last_hidden_state[0].cpu(
            ), tokenizer_result["offset_mapping"][0],
        ):
            offset_mapping = offset_mapping.tolist()
            if offset_mapping[0] >= prev_offset[1]:
                tokens_embeddings[-1].append(embedding)
            else:
                tokens_embeddings.append([embedding])
            prev_offset = offset_mapping

        tokens_embedding = [
            (sum(token_embeddings) / len(token_embeddings)).tolist()
            for token_embeddings in tokens_embeddings
        ]
        return tokens_embedding

    def batch_embed_tokens(self, batch_tokens: List[List[Token]]) -> List[List[Embedding]]:
        tokenizer_result = self.tokenizer.batch_encode_plus(
            batch_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            add_special_tokens=False,
            padding=True,
            truncation=True,
        )

        results = []
        for start_index in range(0, len(tokenizer_result["input_ids"]), self.batch_size):
            end_index = start_index + self.batch_size
            minibatch_input_ids = tokenizer_result["input_ids"][start_index: end_index]
            minibatch_attention_mask = tokenizer_result["attention_mask"][start_index: end_index]
            forward_result = self.model.forward(
                input_ids=minibatch_input_ids.to(self.device),
                attention_mask=minibatch_attention_mask.to(self.device),
            )
            results.append(forward_result)

        batch_model_token_embeddings = [
            last_hidden_state
            for result in results
            for last_hidden_state in result.last_hidden_state.cpu()
        ]

        batch_tokens_embeddings = []
        for model_token_embeddings, model_token_offsets_mapping in zip(
            batch_model_token_embeddings, tokenizer_result["offset_mapping"]
        ):
            prev_offset = [0, 999]
            tokens_embeddings = []
            for embedding, offset_mapping in zip(
                model_token_embeddings, model_token_offsets_mapping
            ):
                offset_mapping = offset_mapping.tolist()
                if offset_mapping == [0, 0]:
                    break

                if offset_mapping[0] >= prev_offset[1]:
                    tokens_embeddings[-1].append(embedding)
                else:
                    tokens_embeddings.append([embedding])
                prev_offset = offset_mapping
            batch_tokens_embeddings.append(tokens_embeddings)

        batch_tokens_embedding = [
            [
                (sum(token_embeddings) / len(token_embeddings)).tolist()
                for token_embeddings in tokens_embeddings
            ] for tokens_embeddings in batch_tokens_embeddings
        ]
        return batch_tokens_embedding

    def embed_sequence(self, sequence: Sequence) -> Embedding:
        tokens = self.tokenizer.encode_plus(
            text=sequence,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        forward_result = self.model.forward(
            input_ids=tokens["input_ids"].to(self.device),
            attention_mask=tokens["attention_mask"].to(self.device),
        )
        return forward_result.pooler_output[0].cpu().tolist()

    def batch_embed_sequence(self, batch_sequence: List[Sequence]) -> Embedding:
        batch_tokens = self.tokenizer.batch_encode_plus(
            batch_sequence,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            padding=True,
            truncation=True,
        )

        results = []
        for start_index in range(0, len(batch_sequence), self.batch_size):
            end_index = start_index + self.batch_size
            minibatch_token_ids = batch_tokens["input_ids"][start_index: end_index]
            minibatch_attention_mask = batch_tokens["attention_mask"][start_index: end_index]
            forward_result = self.model.forward(
                input_ids=minibatch_token_ids.to(self.device),
                attention_mask=minibatch_attention_mask.to(self.device),
            )
            results.append(forward_result)
        result_embeddings = np.empty(
            shape=(len(batch_sequence), results[0].pooler_output.shape[1])
        )
        for result, start_index in zip(results, range(0, len(batch_sequence), self.batch_size)):
            end_index = start_index + self.batch_size
            result_embeddings[start_index: end_index] = result.pooler_output.cpu(
            ).detach().numpy()

        return result_embeddings.tolist()
