import os
import os.path as osp
import sys
import time
import yaml
import imageio
import random
import shutil
import random
import numpy as np
import torch
from tqdm import tqdm

# configuration
class ConfigParser():
    def __init__(self, args):
        """
        class to parse configuration.
        """
        args = args.parse_args()
        self.cfg = self.merge_config_file(args)

        # set random seed
        self.set_seed()

    def __str__(self):
        return str(self.cfg.__dict__)

    def __getattr__(self, name):
        """
        Access items use dot.notation.
        """
        return self.cfg.__dict__[name]

    def __getitem__(self, name):
        """
        Access items like ordinary dict.
        """
        return self.cfg.__dict__[name]

    def merge_config_file(self, args, allow_invalid=True):
        """
        Load json config file and merge the arguments
        """
        assert args.config is not None
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            if 'config' in cfg.keys():
                del cfg['config']
        f.close()
        invalid_args = list(set(cfg.keys()) - set(dir(args)))
        if invalid_args and not allow_invalid:
            raise ValueError(f"Invalid args {invalid_args} in {args.config}.")
        
        for k in list(cfg.keys()):
            if k in args.__dict__.keys() and args.__dict__[k] is not None:
                print('=========>  overwrite config: {} = {}'.format(k, args.__dict__[k]))
                del cfg[k]

        args.__dict__.update(cfg)

        return args

    def set_seed(self):
        ''' set random seed for random, numpy and torch. '''
        if 'seed' not in self.cfg.__dict__.keys():
            return
        if self.cfg.seed is None:
            self.cfg.seed = int(time.time()) % 1000000
        print('=========>  set random seed: {}'.format(self.cfg.seed))
        # fix random seeds for reproducibility
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)

    def save_codes_and_config(self, save_path):
        """
        save codes and config to $save_path.
        """
        cur_codes_path = osp.dirname(osp.dirname(os.path.abspath(__file__)))
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        shutil.copytree(cur_codes_path, osp.join(save_path, 'codes'), \
            ignore=shutil.ignore_patterns('*debug*', '*data*', '*output*', '*exps*', '*.txt', '*.json', '*.mp4', '*.png', '*.jpg', '*.bvh', '*.csv', '*.pth', '*.tar', '*.npz'))

        with open(osp.join(save_path, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(self.cfg.__dict__))
        f.close()


# logger util
class logger:
    """
    Keeps track of the levels and steps of optimization. Logs it via TQDM
    """
    def __init__(self, n_steps, n_lvls):
        self.n_steps = n_steps
        self.n_lvls = n_lvls
        self.lvl = -1
        self.lvl_step = 0
        self.steps = 0
        self.pbar = tqdm(total=self.n_lvls * self.n_steps, desc='Starting')

    def step(self):
        self.pbar.update(1)
        self.steps += 1
        self.lvl_step += 1

    def new_lvl(self):
        self.lvl += 1
        self.lvl_step = 0

    def print(self):
        self.pbar.set_description(f'Lvl {self.lvl}/{self.n_lvls-1}, step {self.lvl_step}/{self.n_steps}')


# other utils
def set_seed(seed=None):
    """
    Set all the seed for the reproducible
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)