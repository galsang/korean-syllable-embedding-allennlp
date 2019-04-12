import argparse
import os

import torch

from allennlp.common.params import Params
from allennlp.common.util import import_submodules
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive

from modules.embedder import SyllableEmbedder
from modules.data import KoWikiReader


class SyllableEmbeddingGenerator(object):
    def __init__(self, vocab: Vocabulary, embedder: SyllableEmbedder) -> None:
        self.vocab = vocab
        self.embedder = embedder
        self.data_reader = KoWikiReader()

    def generate(self, word: str) -> torch.Tensor:
        word = self.data_reader.text_to_instance(source=Token(word))['source']
        word.index(self.vocab)
        word = word.as_tensor(word.get_padding_lengths())['syllables']
        embedding = self.embedder(word)[0]
        return embedding

    def interactive(self) -> None:
        while (1):
            word = input("Enter your word (quit: q):  ")
            if word == 'q':
                exit()
            print(self.generate(word).detach().numpy())

    def convert_file(self,
                     word_input_file_path: str,
                     word_output_file_path: str) -> None:
        with open(word_input_file_path, 'r', encoding='utf-8') as fin:
            with open(word_output_file_path, 'w', encoding='utf-8') as fout:
                for line in fin:
                    word = line.strip()
                    vector = ' '.join([str(v) for v in self.generate(word).detach().numpy()])
                    print(f"{word}\t{vector}", file=fout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, required=True)
    parser.add_argument('--word-input-file-path', type=str)
    parser.add_argument('--word-output-file-path', type=str)
    args = parser.parse_args()

    import_submodules('modules')

    if os.path.exists(args.experiment_path + 'model.tar.gz'):
        archive = load_archive(f'{args.experiment_path}/model.tar.gz')
        model = archive.model
    else:
        config = Params.from_file(f'{args.experiment_path}/config.json')
        config.loading_from_archive = True

        model = Model.load(config.duplicate(),
                           serialization_dir=args.experiment_path,
                           weights_file=f'{args.experiment_path}/best.th')

    vocab = model.vocab
    embedder = model._embedder
    generator = SyllableEmbeddingGenerator(vocab, embedder)
    # generator.interactive()
    if args.word_input_file_path and args.word_output_file_path:
        generator.convert_file(args.word_input_file_path,
                               args.word_output_file_path)


if __name__ == '__main__':
    main()
