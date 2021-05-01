import json
import numpy as np
import time
import os
import sys
import logging
# from yacs.config import CfgNode
import torch
# import atexit
from traceback import print_exception
from datetime import datetime

def get_project_base():
    src_dir = os.path.dirname(os.path.realpath(__file__))
    base = os.path.dirname(src_dir) + "/"
    return base

def neq_load_customized(model, pretrained_dict):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)
    print('---------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model

def track_time(func_name=None):
    def decorater(func):
        if func_name is None:
            name = func.__name__
        else:
            name = func_name
        def wrap(*args, **kwargs):
            start = time.process_time()
            rslt = func(*args, **kwargs)
            end = time.process_time()
            print("%s use %f seconds" % (name, end-start))
            return rslt
        return wrap
    return decorater

def time_str():
    return time.strftime("%m_%d_%H_%M")

def create_logdir(log_dir, create_ck=True, check_exist=True):
    if check_exist and os.path.exists(log_dir):
        print('\nWARNING: log_dir exists %s\n' % log_dir)

    os.makedirs(log_dir, exist_ok=True)
    if create_ck:
        ckpt_dir = os.path.join(log_dir, 'ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)
        return log_dir, ckpt_dir

    return log_dir, None

def create_rerun_script(fname):
    with open(fname, 'w') as fp:
        cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        fp.write( "cd " + os.getcwd() + '\n' )
        fp.write("PY="+sys.executable+'\n')

        if cuda_device:
            cuda_prefix = "CUDA_VISIBLE_DEVICES=%s " % cuda_device
        else:
            cuda_prefix = ""

        fp.write("%s$PY %s\n"%(cuda_prefix, " ".join(sys.argv)))

def prepare_save_env(logdir, exp_name, args=None, create_ck=True, check_exist=True):

    logParentDir = os.path.join(logdir, exp_name)
    logDir, saveDir = create_logdir(logParentDir, create_ck, check_exist)

    rerun_fname = os.path.join(logDir, "run.sh")
    create_rerun_script(rerun_fname)

    if args:
        log_param(print, args)
        argSaveFile = os.path.join(logDir, 'args.json')
        with open(argSaveFile, 'w') as f:
            if not isinstance(args, dict):
                args = vars(args)
            json.dump(args, f, indent=True)


    return logDir, saveDir

def log_param(info, args):
    info('============')

    if not isinstance(args, dict):
        args = vars(args)

    keys = sorted(args.keys())
    for k in keys:
        info( "%s: %s" % (k, args[k]) )

    info('============')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

class Recorder():

    def __init__(self, to_numpy=False):
        self.to_numpy = to_numpy
        self.dict = {}

    def get(self, key):
        if key not in self.dict:
            self.dict[key] = list()
        return self.dict[key]

    def append(self, key, value):
        if self.to_numpy and isinstance(value, torch.Tensor):
            value = to_numpy(value)
        self.get(key).append(value)

    def extend(self, key, values):
        if self.to_numpy:
            good_values = []
            for x in values:
                if isinstance(x, torch.Tensor):
                    x = to_numpy(x)
                good_values.append(x)

        self.get(key).extend(good_values)

    def reset(self, key):
        self.dict[key] = []

    def mean_reset(self, key):
        mean = np.mean(self.get(key))
        self.reset(key)
        return mean

    def get_reset(self, key):
        val = self.get(key)
        self.reset(key)
        return val
