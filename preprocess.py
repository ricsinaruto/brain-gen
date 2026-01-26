import argparse
import yaml

from brain_gen import Omega, MOUS, MOUSConditioned, CamCAN, CamCANConditioned


def main(cli_args=None):
    # parse arguments to script
    dataset_choices = [
        "omega",
        "mous",
        "mous_conditioned",
        "camcan",
        "camcan_conditioned",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--args",
        type=str,
        default="configs/preprocess.yaml",
        help="YAML args file path",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset name",
        required=True,
        choices=dataset_choices,
    )
    parser.add_argument(
        "-s",
        "--stage",
        type=str,
        help="stage of preprocessing",
        required=True,
        choices=["stage_1", "stage_2", "stage_3", "both", "all"],
    )
    script_args = parser.parse_args(cli_args)
    args_file = script_args.args

    # Load args from YAML file
    try:
        with open(args_file, "r") as f:
            args = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Args file not found: {args_file}")

    if script_args.dataset == "omega":
        experiment = Omega(**args)
    elif script_args.dataset == "mous":
        experiment = MOUS(**args)
    elif script_args.dataset == "mous_conditioned":
        experiment = MOUSConditioned(**args)
    elif script_args.dataset == "camcan":
        experiment = CamCAN(**args)
    elif script_args.dataset == "camcan_conditioned":
        experiment = CamCANConditioned(**args)
    else:
        raise ValueError(
            f"Invalid dataset: {script_args.dataset}. "
            f"Choose from: {dataset_choices}"
        )

    if script_args.stage == "stage_1":
        experiment.preprocess_stage_1()
    elif script_args.stage == "stage_2":
        experiment.preprocess_stage_2()
    elif script_args.stage == "stage_3":
        experiment.preprocess_stage_3()
    elif script_args.stage == "both":
        experiment.preprocess_stage_1()
        experiment.preprocess_stage_2()
    elif script_args.stage == "all":
        experiment.preprocess_stage_1()
        experiment.preprocess_stage_2()
        experiment.preprocess_stage_3()
    else:
        raise ValueError(f"Invalid stage: {script_args.stage}")


if __name__ == "__main__":
    main()
