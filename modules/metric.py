import logging
from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import Metric, PearsonCorrelation

from modules.embedder import SyllableEmbedder
from modules.data import KoWikiReader

logger = logging.getLogger(__name__)


@Metric.register("WS353")
class WS353(Metric):
    def __init__(self, sim_file_path: str) -> None:
        self._sim_data = []
        self._sim_gold = []
        self._data_reader = KoWikiReader()
        self._pearson = PearsonCorrelation()

        with open(sim_file_path, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                w1, w2, score = line.strip().split('\t')
                self._sim_data.append((w1, w2))
                self._sim_gold.append(float(score))
        self._sim_gold = torch.tensor(self._sim_gold)

    @overrides
    def __call__(self,
                 vocab: Vocabulary,
                 embedder: SyllableEmbedder,
                 cuda_device: torch.device,
                 print_mode: bool = False) -> None:
        preds = []
        for i in range(len(self._sim_data)):
            w1, w2 = self._sim_data[i]
            w1 = self._data_reader.text_to_instance(source=Token(w1))['source']
            w2 = self._data_reader.text_to_instance(source=Token(w2))['source']

            w1.index(vocab)
            w2.index(vocab)

            w1 = w1.as_tensor(w1.get_padding_lengths())['syllables'].to(cuda_device)
            w2 = w2.as_tensor(w2.get_padding_lengths())['syllables'].to(cuda_device)
            e1, e2 = embedder(w1), embedder(w2)

            preds.append(F.cosine_similarity(e1, e2))

        self._pearson(torch.tensor(preds), self._sim_gold)

        if print_mode:
            print('w1\tw2\tgold\tpred')
            for ((w1, w2), gold, pred) in zip(self._sim_data, self._sim_gold, preds):
                print(f'{w1}\t{w2}\t{gold.item():.2f}\t{pred.item():.2f}')
            print(f'pscore: {self.get_metric():.3f}')

    @overrides
    def get_metric(self, reset: bool = False):
        score = self._pearson.get_metric(reset)
        if reset:
            self.reset()
        return score

    @overrides
    def reset(self):
        self._pearson.reset()
