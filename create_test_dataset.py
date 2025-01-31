# %% [markdown]
# This notebook creates the dataset we used in our analysis. We first filter the
# dataset to remove sentences that can not be scrambled at least 2 levels deep.
# Then we scramble the dataset at different levels of scrambling. We scramble
# the dataset at word level, at an intermediate level (0.5), and at two other
# intermediate levels (1/3 and 2/3).
# %%
import json

import numpy as np
from tqdm import tqdm
from dataset_utils.syntactic_manipulation import is_masked_sentence_deeper_than, scramble_dataset

# %%
# load the flickr30k entities test dataset
annotation_file = "flickr_test_datasets/final_flickr_separateGT_test.json"
with open(annotation_file, "r") as f:
    dataset = json.load(f)
filtered_dataset_file = "flickr_test_datasets/final_flickr_separateGT_test_filtered.json"

# %%
# filter all sentences which can not be shuffled at least 2 levels deep
threshold = 2
excluds = []
for i, sample in enumerate(tqdm(dataset["images"])):
    sentence = sample["caption"]
    phrase_positions = np.array(sample["tokens_positive_eval"]).reshape(-1,2)
    unique_phrases, inverse = np.unique(phrase_positions, axis=0, return_inverse=True)
    if not is_masked_sentence_deeper_than(sentence, unique_phrases, threshold):
        excluds.append(i)
for i in sorted(excluds, reverse=True):
    del dataset["images"][i]
with open(filtered_dataset_file, 'w') as f:
    json.dump(dataset, f)
# %%
# load the filtered dataset
with open(filtered_dataset_file, "r") as f:
    dataset = json.load(f)

# %%
# scramble sentences of the dataset with scrambling at word level
seed = 42
rng = np.random.default_rng(seed)
dataset_scrambled = scramble_dataset(dataset, -1, rng)
dataset_scrambled_file = "flickr_test_datasets/final_flickr_separateGT_test_scrambled_-1.json"
with open(dataset_scrambled_file, 'w') as f:
    json.dump(dataset_scrambled, f)
# %%
# scramble sentences of the dataset with scrambling at an intermediate (0.5) level
with open(filtered_dataset_file, "r") as f:
    dataset = json.load(f)
seed = 43
rng = np.random.default_rng(seed)
dataset_scrambled = scramble_dataset(dataset, level=0.5, rng=rng)
dataset_scrambled_file = "flickr_test_datasets/final_flickr_separateGT_test_scrambled_point5.json"
with open(dataset_scrambled_file, 'w') as f:
    json.dump(dataset_scrambled, f)

# %%
# scramble sentences of the dataset with scrambling at an intermediate (1/3) level
with open(filtered_dataset_file, "r") as f:
    dataset = json.load(f)
seed = 44
rng = np.random.default_rng(seed)
dataset_scrambled = scramble_dataset(dataset, level=1/3, rng=rng)
dataset_scrambled_file = "flickr_test_datasets/final_flickr_separateGT_test_scrambled_point33.json"
with open(dataset_scrambled_file, 'w') as f:
    json.dump(dataset_scrambled, f)

# %%
# scramble sentences of the dataset with scrambling at an intermediate (2/3) level
with open(filtered_dataset_file, "r") as f:
    dataset = json.load(f)
seed = 45
rng = np.random.default_rng(seed)
dataset_scrambled = scramble_dataset(dataset, level=2/3, rng=rng)
dataset_scrambled_file = "flickr_test_datasets/final_flickr_separateGT_test_scrambled_point66.json"
with open(dataset_scrambled_file, 'w') as f:
    json.dump(dataset_scrambled, f)

# %%
# scramble sentences of the dataset with scrambling at word level and also
# scramble words within phrases
with open(filtered_dataset_file, "r") as f:
    dataset = json.load(f)
seed = 46
rng = np.random.default_rng(seed)
dataset_scrambled = scramble_dataset(dataset, level=-1, is_phrase_scrambled=True, rng=rng)
dataset_scrambled_file = "flickr_test_datasets/final_flickr_separateGT_test_scrambled_-1_scramble_within.json"
with open(dataset_scrambled_file, 'w') as f:
    json.dump(dataset_scrambled, f)
