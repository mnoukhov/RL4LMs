import argparse
import os
from argparse import Namespace

import yaml
from haven import haven_wizard as hw

from scripts.training.train_text_generation import main


def run_exp(exp_dict, savedir, args):
    # exp_args = Namespace(**exp_dict)
    # exp_args.saving_dir = savedir

    # if exp_args.wandb:
    #     exp_args.wandb_name = args.exp_group
    #     exp_args.wandb_id = os.path.split(savedir)[-1]
    #     exp_args.wandb_resume = None if args.reset else "allow"
    # wandb.init(name=args.exp_group, config=exp_dict, id=runid, resume=wandb_resume)
    # print(f"running {args.exp_group} with ID {runid}")

    main(
        config_path=args.exp_group,
        project_name="rl4lms",
        experiment_name=os.path.basename(args.exp_group),
        base_path_to_store_results=savedir,
        entity_name="mila-language-drift",
        log_to_wandb=True,
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
        default="/mnt/home/RL4LMs/rl4lm_exps",
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
                f"WANDB_API_KEY={wandb_api_key}",
                f"WANDB_ENTITY=mila-mnoukhov",
                f"WANDB_PROJECT=rl4lms",
            ],
            "restartable": True,
            "resources": {
                "cpu": 4 * args.gpus,
                "mem": 16 * args.gpus,
                "gpu_mem": 16,
                "gpu": args.gpus,
                "gpu_model": "!A100",
            },
            "interactive": False,
            "bid": 9999,
        }
        job_scheduler = "toolkit"
    else:
        job_config = None
        job_scheduler = None

    ## maintain same batch size but account for multi-gpu
    # exp_list = exp_configs.EXP_GROUPS[args.exp_group]
    # for exp_dict in exp_list:
    #     exp_dict["GPU"] = args.gpus
    #     exp_dict["gradient_accumulation_steps"] = int(
    #         exp_dict["gradient_accumulation_steps"] / args.gpus
    #     )

    # For now use 1 config yaml per experiment
    with open(args.exp_group, "r") as fp:
        exp_dict = yaml.safe_load(fp)

    # exp_dict["config_name"] = args.exp_group

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
        # exp_id=args.exp_id,
    )
