import argparse
import yaml
import json
import torch
import mne

mne.set_log_level("WARNING")
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from brain_gen.eval.eval_runner import EvaluationRunner  # noqa: E402


def main(cli_args=None):
    # parse arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", type=str, help="args file name", required=True)
    parser.add_argument(
        "-d", "--dict", action="store_true", help="dict mode", default=False
    )

    script_args = parser.parse_args(cli_args)
    args_file = script_args.args

    if script_args.dict:
        # convert json string to dict using json.loads
        cfg = json.loads(args_file)
    else:
        with open(args_file) as f:
            cfg = yaml.safe_load(f)

    device = None
    if isinstance(cfg, dict):
        device = cfg.get("eval_runner", {}).get("device")
    EvaluationRunner(cfg, device=device).run()


if __name__ == "__main__":
    main()
