import argparse
import os
from tensorboardX import SummaryWriter

import torch

from allennlp.common.params import Params
from allennlp.common.util import import_submodules
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive

from modules.embedder import SyllableEmbedder
from modules.data import KoWikiReader
from modules.metric import WS353


class Visualization(object):
    def __init__(self, experiment_path: str,
                 vocab: Vocabulary,
                 embedder: SyllableEmbedder,
                 embedding_dim: int = 300) -> None:
        summary_writer = SummaryWriter(f'{experiment_path}/log/visualization')
        data_reader = KoWikiReader()

        words = [vocab.get_token_from_index(i, namespace='words')
                 for i in range(vocab.get_vocab_size('words'))]
        embeddings = torch.zeros(vocab.get_vocab_size('words'), embedding_dim)

        for i, c in enumerate(words):
            word = data_reader.text_to_instance(source=Token(c))['source']
            word.index(vocab)
            word_tensor = word.as_tensor(word.get_padding_lengths())['syllables']
            embeddings[i] = embedder(word_tensor)

        summary_writer.add_embedding(embeddings, metadata=words, tag='syllable_embeddings')
        summary_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, required=True)
    parser.add_argument('--ws353-path', type=str, required=True)
    parser.add_argument('--cuda-device', type=int, default=-1)
    args = parser.parse_args()
    args.cuda_device = f'cuda:{cuda_device}' \
        if torch.cuda.is_available() and args.cuda_device >= 0 else 'cpu'

    import_submodules('modules')

    if os.path.isfile(f'{args.experiment_path}/model.tar.gz'):
        archive = load_archive(f'{args.experiment_path}/model.tar.gz')
        model = archive.model
    else:
        config = Params.from_file(f'{args.experiment_path}/config.json')
        config.loading_from_archive = True

        model = Model.load(config.duplicate(),
                           serialization_dir=args.experiment_path,
                           weights_file=f'{args.experiment_path}/best.th')

    metric = WS353(args.ws353_path)
    vocab = model.vocab
    embedder = model.embedder
    metric(vocab, embedder, args.cuda_device, print_mode=True)
    if not os.path.exists(f'{args.experiment_path}/log/visualization'):
        Visualization(args.experiment_path, vocab, embedder)


if __name__ == '__main__':
    main()
