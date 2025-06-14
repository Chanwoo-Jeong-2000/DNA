from SELFRec import SELFRec
from util.conf import ModelConf
import time
import argparse
import random
import numpy as np
import torch

seed = 2020
deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def print_models(title, models):
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default=None, help='Path to model YAML config file')
    args = parser.parse_args()
    
    models = {
        'Graph-Based Baseline Models': ['LightGCN', 'DirectAU', 'MF'],
        'Self-Supervised Graph-Based Models': ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF'],
        'Sequential Baseline Models': ['SASRec'],
        'Self-Supervised Sequential Models': ['CL4SRec', 'BERT4Rec']
    }

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print_models("Available Models", models)

    #model = input('Please enter the model you want to run:')
    model = "XSimGCL"
    config_path = args.conf_path# or './conf/XSimGCL_ori.yaml'



    s = time.time()
    all_models = sum(models.values(), [])
    if model in all_models:
        conf = ModelConf(config_path)
        rec = SELFRec(conf)
        rec.execute()
        e = time.time()
        print(f"Running time: {e - s:.2f} s")
    else:
        print('Wrong model name!')
        exit(-1)
