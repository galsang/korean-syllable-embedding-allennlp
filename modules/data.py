from typing import List
import logging

from overrides import overrides

import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register('KoWiki')
class KoWikiReader(DatasetReader):
    def __init__(self,
                 window_size: int = 4,
                 min_padding_length: int = 4,
                 subsampling_threshold: float = 10e-5,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._window_size = window_size
        self._subsampling_threshold = subsampling_threshold
        self._word_indexers = {'words': SingleIdTokenIndexer(namespace='words')}
        self._syllable_indexers = {
            'syllables': TokenCharactersIndexer(
                namespace='syllables', min_padding_length=min_padding_length)}
        self._word_sample_prob = None

    @overrides
    def _read(self, file_path):
        # we need to prepare word frequency stats before dealing with our corpus to subsample frequent words
        if self._word_sample_prob is None:
            logger.info(f'Building word frequency stats...')
            self._word_sample_prob = {}
            total = 0
            with open(cached_path(file_path), 'r') as f:
                for line in Tqdm.tqdm(f.readlines()):
                    tokens = line.strip().split()
                    for token in tokens:
                        if token in self._word_sample_prob:
                            self._word_sample_prob[token] += 1
                        else:
                            self._word_sample_prob[token] = 1
                        total += 1

                for k, v in self._word_sample_prob.items():
                    # convert count into frequency
                    self._word_sample_prob[k] = v / total
                    # word downsampling to prevent frequent words from being shown so much
                    self._word_sample_prob[k] = max(0, 1 - np.sqrt(self._subsampling_threshold /
                                                                   self._word_sample_prob[k]))

        logger.info(f'Reading instances from lines in file at {file_path}')
        with open(cached_path(file_path), 'r') as f:
            for line in Tqdm.tqdm(f.readlines()):
                tokens = line.strip().split()
                for i in range(len(tokens)):
                    if np.random.binomial(1, self._word_sample_prob[tokens[i]]):
                        start = max(0, i - self._window_size)
                        end = min(len(tokens) - 1, i + self._window_size)

                        source = Token(tokens[i])
                        targets = [Token(tokens[j]) for j in range(start, end + 1) if i != j]

                        yield self.text_to_instance(source, targets)

    @overrides
    def text_to_instance(self, source: Token, targets: List[Token] = []) -> Instance:
        fields = {'source': TextField([source], self._syllable_indexers),
                  'targets': TextField(targets, self._word_indexers)}
        return Instance(fields)
