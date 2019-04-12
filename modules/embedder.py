from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import TokenEmbedder, Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, CnnEncoder


@TokenEmbedder.register("syllable")
class SyllableEmbedder(TokenEmbedder):
    def __init__(self,
                 syllable_embedding: Embedding,
                 syllable_encoder: CnnEncoder,
                 dropout: float = 0.0) -> None:
        super(TokenEmbedder, self).__init__()

        self._embedding = syllable_embedding
        self._encoder = syllable_encoder
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    @overrides
    def get_output_dim(self) -> int:
        return self._encoder.get_output_dim()  # pylint: disable=protected-access

    @overrides
    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        mask = (token_characters != 0).long()
        return self._dropout(self._encoder(self._embedding(token_characters), mask))

    # The setdefault requires a custom from_params
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SyllableEmbedder':  # type: ignore
        # pylint: disable=arguments-differ
        embedding_params: Params = params.pop("syllable_embedding")
        # Embedding.from_params() uses "tokens" as the default namespace, but we need to change
        # that to be "token_characters" by default. If num_embeddings is present, set default namespace
        # to None so that extend_vocab call doesn't misinterpret that some namespace was originally used.
        default_namespace = None if embedding_params.get("num_embeddings", None) else "token_characters"
        embedding_params.setdefault("vocab_namespace", default_namespace)
        embedding = Embedding.from_params(vocab, embedding_params)
        encoder_params: Params = params.pop("syllable_encoder")
        encoder = Seq2VecEncoder.from_params(encoder_params)
        dropout = params.pop_float("dropout", 0.0)
        params.assert_empty(cls.__name__)
        return cls(embedding, encoder, dropout)
