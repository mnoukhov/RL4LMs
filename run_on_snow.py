import argparse
import os

import yaml
from haven import haven_wizard as hw

from scripts.training.train_text_generation import main


def run_exp(exp_dict, savedir, args):
    main(
        config_path=args.exp_group,
        project_name="rl4lms",
        experiment_name=os.path.basename(args.exp_group),
        base_path_to_store_results=savedir,
        entity_name="mila-language-drift",
        log_to_wandb=(args.job_scheduler is not None),
    )


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="/mnt/home/results/rl4lms",
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-r",
        "--reset",
        type=int,
        default=0,
        help="If true, reset the experiment. Else, resume.",
    )
    parser.add_argument(
        "-j",
        "--job_scheduler",
        default=None,
        type=str,
        help="Run the experiments as jobs in the cluster.",
    )
    parser.add_argument(
        "-p",
        "--python_binary",
        default="/mnt/home/miniconda/envs/slot_filling/bin/python",
        help="path to your python executable",
    )
    parser.add_argument(
        "-n", "--gpus", default=1, type=int, help="number of gpus to use for experiment"
    )
    # parser.add_argument(
    #     "--exp-id", default=None, help="id used to resume an experiment"
    # )

    args, _ = parser.parse_known_args()

    if args.job_scheduler == "toolkit":
        with open("/mnt/home/wandb_api_key", "r") as f:
            wandb_api_key = f.read().rstrip()

        job_config = {
            "account_id": os.environ["EAI_ACCOUNT_ID"],
            "image": "registry.console.elementai.com/snow.colab/cuda",
            "data": [
                "snow.mnoukhov.home:/mnt/home",
                "snow.colab.public:/mnt/public",
            ],
            "environment_vars": [
                f"HF_HOME=/mnt/public/datasets/huggingface/",
                f"WANDB_API_KEY={wandb_api_key}",
            ],
            "restartable": True,
            "resources": {
                "cpu": 4 * args.gpus,
                "mem": 16 * args.gpus,
                "gpu_mem": 80,
                "gpu": args.gpus,
                "gpu_model": "A100",
            },
            "interactive": False,
            "bid": 9999,
        }
        job_scheduler = "toolkit"
    else:
        job_config = None
        job_scheduler = None

    # For now use 1 config yaml per experiment
    with open(args.exp_group, "r") as fp:
        exp_dict = yaml.safe_load(fp)

    # Run experiments and create results file
    hw.run_wizard(
        func=run_exp,
        exp_list=[exp_dict],
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        job_scheduler=job_scheduler,
        results_fname="rl4lm_exps/notebook.ipynb",
        python_binary_path=args.python_binary,
        args=args,
        use_threads=True,
        save_logs=False,
        # exp_id=args.exp_id,
    )
