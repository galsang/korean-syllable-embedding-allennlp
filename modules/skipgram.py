from typing import Dict

import logging
from overrides import overrides

import torch
import torch.nn as nn

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import TokenEmbedder

from modules.metric import WS353

logger = logging.getLogger(__name__)
VERY_SMALL_NUMBER = 1e-30


@Model.register('SGNS')
class SkipGramWithNegativeSampling(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TokenEmbedder,
                 sim_file_path: str,
                 window_size: int = 4,
                 num_neg_samples: int = 5,
                 neg_exponent: float = 0.75,
                 cuda_device: int = -1) -> None:
        super().__init__(vocab)
        self.device = f'cuda:{cuda_device}' if torch.cuda.is_available() and cuda_device >= 0 else 'cpu'
        self.ws353 = WS353(sim_file_path)

        self.embedder = embedder
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.output_layer = nn.Linear(embedder.get_output_dim(),
                                      vocab.get_vocab_size('words'))

        # negative sampling with word frequency distribution
        self.word_dist = torch.zeros(vocab.get_vocab_size('words'))
        if vocab._retained_counter:
            for word, count in vocab._retained_counter['words'].items():
                word_idx = vocab.get_token_index(token=word, namespace='words')
                self.word_dist[word_idx] = count
            # prevent sampling process from choosing pad and unk tokens
            self.word_dist[vocab.get_token_index(token=vocab._padding_token, namespace='words')] = 0
            self.word_dist[vocab.get_token_index(token=vocab._oov_token, namespace='words')] = 0
            # prevent frequent words from sampling too frequently
            self.word_dist = torch.pow(self.word_dist, neg_exponent)

    @overrides
    def forward(self,
                source: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, context_size = targets['words'].size()
        # (batch_size, num_syllables) -> (batch_size, emb_dim)
        embedding = self.embedder(source['syllables'].squeeze())
        # (batch_size, vocab_size)
        pred = self.output_layer(embedding)
        loss = self.loss(batch_size, context_size, pred, targets)
        self.ws353(self.vocab, self.embedder, self.device)
        output_dict = {'loss': loss}
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'WS353_S': self.ws353.get_metric(reset)}
        return metrics

    def sample(self, batch_size: int, context_size: int) -> torch.Tensor:
        # (batch_size * context_size * num_neg_samples)
        neg_samples = torch.multinomial(
            self.word_dist,
            batch_size * context_size * self.num_neg_samples,
            replacement=True)
        return neg_samples

    def loss(self, batch_size: int, context_size: int,
             pred: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # (batch_size * context_size)
        pos_samples = targets['words'].view(-1)
        # (batch_size * context_size * num_neg_samples)
        neg_samples = self.sample(batch_size, context_size)
        # (batch_size * context_size)
        pos_indexer = [i for i in range(batch_size) for _ in range(context_size)]
        pos_loss = (pred[pos_indexer, pos_samples].sigmoid() + VERY_SMALL_NUMBER).log().sum()
        # repeat by the number of negative examples: (batch_size * context_size * num_neg_examples)
        neg_indexer = pos_indexer * self.num_neg_samples
        neg_loss = (1 - pred[neg_indexer, neg_samples].sigmoid() + VERY_SMALL_NUMBER).log().sum()

        return - (pos_loss + neg_loss) / (batch_size + VERY_SMALL_NUMBER)
