import argparse
import copy
import logging
import time

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.training.util import time_to_str


def train(args):
    params = Params.from_file(args.param_path, args.overrides)
    _params = copy.deepcopy(params)
    model = train_model(params,
                        args.serialization_dir,
                        args.file_friendly_logging,
                        args.recover,
                        args.force)

    return model, _params


def main():
    LEVEL = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)
    start_time = time_to_str(time.time())
    experiment_dir = f'experiments/{start_time}'

    parser = argparse.ArgumentParser()
    parser.add_argument('--param-path',
                        type=str,
                        default='settings.jsonnet',
                        help='path to parameter file describing the model to be trained')
    parser.add_argument('-s', '--serialization-dir',
                        type=str,
                        default=experiment_dir,
                        help='directory in which to save the model and its logs')
    parser.add_argument('-r', '--recover',
                        action='store_true',
                        default=False,
                        help='recover training from the state in serialization_dir')
    parser.add_argument('-f', '--force',
                        action='store_true',
                        required=False,
                        help='overwrite the output directory if it exists')
    parser.add_argument('-o', '--overrides',
                        type=str,
                        default="",
                        help='a JSON structure used to override the experiment configuration')
    parser.add_argument('--file-friendly-logging',
                        action='store_true',
                        default=False,
                        help='outputs tqdm status on separate lines and slows tqdm refresh rate')
    args = parser.parse_args()
    import_submodules('modules')
    model, params = train(args)


if __name__ == '__main__':
    main()
