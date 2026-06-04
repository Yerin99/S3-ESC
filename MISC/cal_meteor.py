
# from flow_score import *
from nltk.translate.meteor_score import meteor_score
import torch
import json

def prepare_data(ref_file, hyp_file):
    result = []
    scores = []
    with open(ref_file, 'r', encoding='utf-8') as f:
        refs = json.load(f)
    with open(hyp_file, 'r', encoding='utf-8') as f:
        hyps = json.load(f)
    for ref, hyp in zip(refs, hyps):
        score = meteor_score([ref], hyp)
        scores.append(score)

    return scores

# torch.nn.Module.dump_patches = True
import numpy as np
# MODEL_PATH = "models/DialoFlow_large.bin"
hypFile = 'generated_data/all_loss/hyp_strategy.json'
refFile = 'generated_data/all_loss/ref_strategy.json'
result = prepare_data(refFile, hypFile)
result = np.array(result)
print(result.mean(0))


