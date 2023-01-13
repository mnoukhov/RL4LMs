import argparse
import copy
import os

import yaml
from haven import haven_utils as hu
from haven import haven_wizard as hw

from scripts.training.train_text_generation import main


def run_exp(exp_dict, savedir, args):
    main(
        config_path=args.exp_group,
        project_name="rl4lms",
        experiment_name=os.path.basename(args.exp_group),
        base_path_to_store_results=savedir,
        entity_name="mila-language-drift",
        log_to_wandb=(not args.no_wandb),
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
        default="/home/toolkit/RL4LMs/results",
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
        default="/home/toolkit/.conda/envs/rl4lms/bin/python",
        help="path to your python executable",
    )
    parser.add_argument(
        "-n", "--gpus", default=1, type=int, help="number of gpus to use for experiment"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="disable wandb", default=False
    )
    parser.add_argument("--seeds", type=int, default=1)
    # parser.add_argument(
    #     "--exp-id", default=None, help="id used to resume an experiment"
    # )

    args, _ = parser.parse_known_args()

    # 1 config yaml per experiment
    with open(args.exp_group, "r") as fp:
        exp_dict = yaml.safe_load(fp)

    exp_list = []
    for seed in range(args.seeds):
        seed_exp_dict = copy.deepcopy(exp_dict)
        seed_exp_dict["alg"]["args"]["seed"] = seed
        exp_list.append(seed_exp_dict)

    if args.job_scheduler == "toolkit":
        with open("/home/toolkit/wandb_api_key", "r") as f:
            wandb_api_key = f.read().rstrip()

        job_config = {
            "account_id": os.environ["EAI_ACCOUNT_ID"],
            # "image": "registry.console.elementai.com/snow.colab/cuda",
            # "image": "registry.console.elementai.com/snow.colab_public/ssh",
            # "image": "registry.console.elementai.com/snow.mnoukhov/rl4lms",
            "image": "registry.console.elementai.com/snow.interactive_toolkit/default",
            "data": [
                "snow.mnoukhov.home:/home/toolkit",
                "snow.colab.public:/mnt/public",
            ],
            "environment_vars": [
                "HOME=/home/toolkit",
                f"HF_HOME=/home/toolkit/huggingface/",
                f"WANDB_API_KEY={wandb_api_key}",
                f"WANDB_RUN_GROUP={hu.hash_dict(exp_dict)}",
            ],
            "restartable": True,
            "resources": {
                "cpu": 4 * args.gpus,
                "mem": 48 * args.gpus,
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

    # Run experiments and create results file
    hw.run_wizard(
        func=run_exp,
        exp_list=exp_list,
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        job_scheduler=job_scheduler,
        results_fname="results/notebook.ipynb",
        python_binary_path=args.python_binary,
        args=args,
        use_threads=True,
        save_logs=False,
        # exp_id=args.exp_id,
    )
