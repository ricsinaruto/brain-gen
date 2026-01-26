import argparse
import yaml

import torch
import mne

mne.set_log_level("WARNING")
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

from brain_gen import (  # noqa: E402
    ExperimentTokenizer,
    ExperimentDL,
    ExperimentTokenizerText,
    ExperimentVidtok,
)


def main(cli_args=None):
    # parse arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", type=str, help="args file name", required=True)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="run mode",
        required=True,
        choices=[
            "train",
            "test",
            "eval",
            "tokenizer",
            "tokenizer-text",
            "vidtok",
        ],
    )

    script_args = parser.parse_args(cli_args)
    args_file = script_args.args
    mode = script_args.mode

    with open(args_file) as f:
        cfg = yaml.safe_load(f)

    if mode == "train":
        ExperimentDL(cfg).train()
    elif mode == "test":
        ExperimentDL(cfg).test()
    elif mode == "tokenizer":
        ExperimentTokenizer(cfg)
    elif mode == "tokenizer-text":
        ExperimentTokenizerText(cfg)
    elif mode == "vidtok":
        ExperimentVidtok(cfg).train()
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
