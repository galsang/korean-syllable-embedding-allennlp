local data_path = "data/ko_wiki.txt";
local sim_file_path = "data/ws353_s_sh.txt";
local seed = 123;
local cuda_device = 0;
local num_workers = 8;

local batch_size = 4096;
local optimizer = "adam";
local learning_rate = 0.001;
local num_epochs = 12;

local syllable_dim = 25;
local num_filters = 75;
local ngram_filter_sizes = [1,2,3,4];
local embedding_dim = 300;
local max_vocab_size = 10000;
local min_word_cnt = 5;
local neg_exponent = 0.75;
local subsampling_threshold = 10e-5;
local num_neg_samples = 7;
local window_size = 4;

{
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed,
    "train_data_path": data_path,
    "dataset_reader": {
        "type": "multiprocess",
        "num_workers": num_workers,
        "base_reader" : {
            "type": "KoWiki",
            "window_size": window_size,
            "min_padding_length": 4,
            "subsampling_threshold": subsampling_threshold,
            "lazy": true
        }
    },
    "model": {
        "type": "SGNS",
        "window_size": window_size,
        "num_neg_samples": num_neg_samples,
        "neg_exponent": neg_exponent,
        "sim_file_path": sim_file_path,
        "cuda_device": cuda_device,
        "embedder": {
            "type": "syllable",
            "syllable_embedding": {
                "vocab_namespace": "syllables",
                "embedding_dim": syllable_dim
            },
            "syllable_encoder": {
                "type": "cnn",
                "embedding_dim": syllable_dim,
                "num_filters": num_filters,
                "ngram_filter_sizes": ngram_filter_sizes,
                "conv_layer_activation": "tanh"
            },
            "dropout": 0.0
        }
    },
    "iterator": {
        "type": "multiprocess",
        "num_workers": num_workers,
        "base_iterator": {
            "type": "bucket",
            "batch_size": batch_size,
            "sorting_keys": [["source", "num_token_characters"]]
        }
    },
    "trainer": {
        "cuda_device": cuda_device,
        "num_epochs": num_epochs,
        "optimizer": {
            "type": optimizer,
            "lr": learning_rate
        },
    },
    "vocabulary": {
        "min_count": {
            "words": min_word_cnt
        },
        "max_vocab_size": {
            "words": max_vocab_size
        }
    }
}
