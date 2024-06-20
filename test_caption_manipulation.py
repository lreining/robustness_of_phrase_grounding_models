# %%
from syntactic_manipulation import ScrambledSentence
from dataset_utils.ModulatedDetection import ModulatedDetection
from transformers import RobertaTokenizerFast

import torch_geometric.transforms as transforms
import torch
import numpy as np
# %%
# load dataset
image_path = "flickr30k_entities/flickr30k-images"
ann_file = "final_flickr_mergedGT_test.json"
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
dataset = ModulatedDetection(image_path, ann_file, None, True, tokenizer, False)
# %%
tsfm = ScrambledSentence(-1,1)
dataset2 = tsfm(dataset)
