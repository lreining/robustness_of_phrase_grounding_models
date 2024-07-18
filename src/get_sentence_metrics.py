import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset_utils.syntactic_tree import SyntacticTree
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance


def compute_distances_for_scrambled_sentences(path=None):
    """
    Computes the normalized Damerau-Levenshtein distances between original captions and manipulated captions
    for a set of scrambled sentences.

    Parameters:
    - path (str, optional): The path to save the output CSV file. Defaults to None.

    Returns:
    - None
    """
    
    manipulations = ["-1", "point66", "point5", "point33"]
    manipulated_file_names = [f"../flickr_test_datasets/final_flickr_separateGT_test_scrambled_{manipulation}.json" for manipulation in manipulations]
    original_file_name = f"../flickr_test_datasets/final_flickr_separateGT_test_filtered.json"

    original_file = open(original_file_name,"r")
    manipulated_files = [open(file,"r") for file in manipulated_file_names]

    manipulated_datasets = [json.load(file) for file in manipulated_files]
    original_dataset = json.load(original_file)

    values = []

    for i, sample in tqdm(enumerate(original_dataset["images"])):
        original_caption = sample["caption"].replace(" .", "").lower()
        img_id = sample["original_img_id"]
        sentence_id = sample["sentence_id"]
        value = ["_".join([str(img_id), str(sentence_id)])]
        for manipulated_dataset in manipulated_datasets:
            manipulated_caption = manipulated_dataset["images"][i]["caption"]
            distance = normalized_damerau_levenshtein_distance(original_caption, manipulated_caption)
            value += [distance]
        values.append(value)

    df = pd.DataFrame(columns=["Image_Id"]+[f"DL_Dist_{manipulation}" for manipulation in manipulations], data=values)
    df = df.set_index("Image_Id")
    path = f"{path}/" if path is not None else ""
    df.to_csv(f"{path}flickr_test_datasets_sentence_metrics/normalized_damerau_levenshtein_distances.csv")

    for f in [original_file] + manipulated_files:
        f.close()

def dataset_summary(dataset_name, path=None):
    """
    Generates a summary of sentence metrics for a given dataset.

    Parameters:
    - dataset_name (str): The name of the dataset.
    - path (str, optional): The path to save the summary CSV file. Defaults to None.

    Returns:
    - None
    """
    file = f"../flickr_test_datasets/final_flickr_separateGT_test_{dataset_name}.json"
    with open(file, "r") as f:
        dataset = json.load(f)
        values = []
        for sample in tqdm(dataset["images"]):
            caption = sample["caption"].replace(" .", "").lower()
            img_id = sample["original_img_id"]
            sentence_id = sample["sentence_id"]
            tree = SyntacticTree(caption)
            widths = tree.get_widths()
            heights = tree.get_heights()
            values.append(["_".join([str(img_id), str(sentence_id)]), np.max(widths), np.mean(widths), np.max(heights), np.mean(heights)])
        df = pd.DataFrame(columns=["Image_Id", "Max_Width", "Mean_Width", "Max_Height", "Mean_Height"],data=values)
        df = df.set_index("Image_Id")
        path = f"{path}/" if path is not None else ""
        df.to_csv(f"{path}flickr_test_datasets_sentence_metrics/final_flickr_separateGT_test_{dataset_name}_summary.csv")





