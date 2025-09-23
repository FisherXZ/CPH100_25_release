import argparse
import json
import random
import os, subprocess
from csv import DictWriter
import multiprocessing
import itertools
import sys
from typing import List

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config_path",
        type=str,
        default="grid_search.json",
        help="Location of config file"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of processes to run in parallel"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Location of experiment logs and results"
    )

    parser.add_argument(
        "--grid_search_results_path",
        default="grid_results.csv",
        help="Where to save grid search results"
    )

    return parser

def get_experiment_list(config: dict) -> List[dict]:
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item, 
    but the config contains a list, this will return one job for each item in the list.
    
    Args:
        config: experiment configuration dictionary from grid_search.json
        
    Example config:
    {
        "learning_rate": [0.001, 0.01],
        "batch_size": [64, 128],
        "regularization_lambda": [0, 0.1]
    }
    
    Returns:
        jobs: a list of dicts, each of which encapsulates one job.
        Example: [
            {"learning_rate": 0.001, "batch_size": 64, "regularization_lambda": 0},
            {"learning_rate": 0.001, "batch_size": 64, "regularization_lambda": 0.1},
            {"learning_rate": 0.001, "batch_size": 128, "regularization_lambda": 0},
            ...
        ]
    '''
    # Get all parameter names and their possible values
    param_names = list(config.keys())
    param_values = [config[param] for param in param_names]
    
    # Generate all combinations using itertools.product
    jobs = []
    for combination in itertools.product(*param_values):
        # Create a job dict by pairing parameter names with values
        job = dict(zip(param_names, combination))
        jobs.append(job)
    
    return jobs

def worker(args: argparse.Namespace, job_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue):
    '''
    Worker thread for each worker. Consumes all jobs and pushes results to done_queue.
    :args - command line args
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, params))

def launch_experiment(args: argparse.Namespace, experiment_config: dict) -> dict:
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    
    Args:
        args: command line arguments
        experiment_config: flags to use for this model run. Will be fed into main.py
        
    Returns:
        dict: flags for this experiment as well as result metrics
        
    Example return:
    {
        "learning_rate": 0.001,
        "batch_size": 64, 
        "regularization_lambda": 0.1,
        "train_auc": 0.65,
        "val_auc": 0.62
    }
    '''

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # Create unique filename for this experiment
    experiment_id = "_".join([f"{k}_{v}" for k, v in experiment_config.items()])
    log_file = os.path.join(args.log_dir, f"experiment_{experiment_id}.log")
    results_file = os.path.join(args.log_dir, f"results_{experiment_id}.json")
    
    # Build command to run main.py with these parameters
    cmd = [
        sys.executable, "main.py",
        "--results_path", results_file
    ]
    
    # Add each parameter from experiment_config to command
    for param_name, param_value in experiment_config.items():
        cmd.extend([f"--{param_name}", str(param_value)])
    
    # Run the experiment and capture output
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=f, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            print(f"Experiment failed: {experiment_config}")
            return {"error": "experiment_failed", **experiment_config}
        
        # Load results from the output file
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {"error": "no_results_file"}
        
        # Combine experiment parameters with results
        final_results = {**experiment_config, **results}
        return final_results
        
    except Exception as e:
        print(f"Error running experiment {experiment_config}: {e}")
        return {"error": str(e), **experiment_config}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> List[dict]:
    print(args)
    config = json.load(open(args.config_path, "r"))
    print("Starting grid search with the following config:")
    print(json.dumps(config, indent=2))

    # TODO: From config, generate a list of experiments to run
    experiments = get_experiment_list(config)
    random.shuffle(experiments)

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for exper in experiments:
        job_queue.put(exper)

    print("Launching dispatcher with {} experiments and {} workers".format(len(experiments), args.num_workers))

    # TODO: Define worker fn to launch an experiment as a separate process.
    for _ in range(args.num_workers):
        multiprocessing.Process(target=worker, args=(args, job_queue, done_queue)).start()

    # Accumualte results into a list of dicts
    grid_search_results = []
    for _ in range(len(experiments)):
        grid_search_results.append(done_queue.get())

    keys = grid_search_results[0].keys()

    print("Saving results to {}".format(args.grid_search_results_path))

    writer = DictWriter(open(args.grid_search_results_path, 'w'), keys)
    writer.writeheader()
    writer.writerows(grid_search_results)

    print("Done")
    return grid_search_results

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)