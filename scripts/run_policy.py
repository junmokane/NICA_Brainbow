from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
import numpy as np
import matplotlib.pyplot as plt
from rlkit.core import logger


filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    num_fail = 0
    for _ in range(args.ep):
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
            sleep=args.S,
        )
        if np.any(path['rewards'] == -1):
            num_fail += 1
            if args.de:
                last_obs = np.moveaxis(np.reshape(path['observations'][-1], (3, 33, 33)), 0, -1)
                last_next_obs = np.moveaxis(np.reshape(path['next_observations'][-1], (3, 33, 33)), 0, -1)
                last_obs = (last_obs * 33 + 128).astype(np.uint8)
                last_next_obs = (last_next_obs * 33 + 128).astype(np.uint8)
                fig = plt.figure(figsize=(10, 10))
                fig.add_subplot(2, 1, 1)
                plt.imshow(last_obs)
                fig.add_subplot(2, 1, 2)
                plt.imshow(last_next_obs)
                plt.show()
                plt.close()

        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

    print('number of failures:', num_fail)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--de', type=bool, default=False,
                        help='stop and detect failure case.')
    parser.add_argument('--ep', type=int, default=1000,
                        help='# of episodes to run')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--S', type=float, default=1,
                        help='time sleep when rendering')
    parser.add_argument('--gpu', action='store_false')
    args = parser.parse_args()

    simulate_policy(args)
