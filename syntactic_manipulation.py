from tqdm import tqdm
import numpy as np
from dataset_utils.syntactic_tree import SyntacticTree
from copy import deepcopy

MASK = "XXXX"

def scramble_sentence(sentence, level,rng=None):
    if not rng:
        rng = np.random.default_rng()
    tree = SyntacticTree(sentence)
    if level < 1 and level > 0:
        level = np.ceil(tree.get_max_height()*level)
    subsentences, _ = tree.get_sentence_parts_for_level(level)
    rng.shuffle(subsentences)
    return subsentences


def scramble_phrase(sentence,level, phrase_idx_bounds, rng=None):
    if not rng:
        rng = np.random.default_rng()
    tree = SyntacticTree(sentence)
    if level < 1 and level > 0:
        level = np.ceil(tree.get_max_height()*level)
    sub_sentences,_ = tree.get_sentence_parts_for_level(level)
    sub_phrases = []
    idx = 0
    for sub_sentence in sub_sentences:
        if idx >= phrase_idx_bounds[0] and idx < phrase_idx_bounds[1]:
            sub_phrases.append(sub_sentence)
        idx += len(sub_sentence)+1
    rng.shuffle(sub_phrases)
    return sub_phrases

def scramble(sentence, phrase_idx_bounds, level, rng=None, is_phrase_scrambled=False):
    if not rng:
        rng = np.random.default_rng()
    phrase_idx_bounds = phrase_idx_bounds[phrase_idx_bounds[:,0].argsort()]
    masked_sentence = sentence
    decrement = 0
    for i,idx in enumerate(phrase_idx_bounds):
        mask = "".join([MASK,str(i)])
        masked_sentence = mask_phrase(masked_sentence, [idx[0] - decrement, idx[1] - decrement], mask)
        decrement += idx[1]-idx[0]-len(mask)
    sub_sentences = scramble_sentence(masked_sentence, level, rng)
    if is_phrase_scrambled:
        sub_phrases = []
        for idx in phrase_idx_bounds:
            sub_phrase = " ".join(scramble_phrase(sentence, level, idx, rng))
            sub_phrase = sentence[idx[0]:idx[1]] if len(sub_phrase)==0 else sub_phrase
            sub_phrases.append(sub_phrase)
    else:
        sub_phrases = [sentence[idx[0]:idx[1]] for idx in phrase_idx_bounds]
    sub_sentences = " ".join(sub_sentences)
    return unmask_phrases(sub_sentences, sub_phrases)


def mask_phrase(sentence, phrase_idx_bounds,mask=None):
    mask = MASK if mask is None else mask
    start_idx, end_idx = phrase_idx_bounds
    sentence = "".join([sentence[:start_idx], mask, sentence[end_idx:]])
    return sentence

def unmask_phrases(sentence, phrases,mask=None):
    mask = MASK if mask is None else mask
    idx = sentence.find(mask)
    while idx != -1:
        complete_mask = sentence[idx:].split(" ")[0]

        phrase_idx = int(complete_mask.replace(mask,""))
        phrase = phrases[phrase_idx]
        sentence = "".join([sentence[:idx], phrase, sentence[idx+len(complete_mask):]])
        phrases[phrase_idx] = [idx, idx+len(phrase)]

        idx = sentence.find(mask)
    return sentence.lower(), phrases

def is_masked_sentence_deeper_than(sentence, phrase_idx_bounds, threshold):
    phrase_idx_bounds = phrase_idx_bounds[phrase_idx_bounds[:,0].argsort()]
    masked_sentence = sentence
    decrement = 0
    for i,idx in enumerate(phrase_idx_bounds):
        mask = "".join([MASK,str(i)])
        masked_sentence = mask_phrase(masked_sentence, [idx[0] - decrement, idx[1] - decrement], mask)
        decrement += idx[1]-idx[0]-len(mask)
    tree = SyntacticTree(masked_sentence)
    if tree.get_max_height() > threshold:
        return True
    else:
        return False


def scramble_dataset(dataset, level, rng=None):
    """
    Scrambles the captions in the given dataset according to the specified level
    of scrambling and returns a copy of the dataset in which the captions are
    scrambled.

    Parameters:
    - dataset (list of dict): The dataset to be scrambled. Each element in the list is a dictionary representing
      a data sample, which must include a 'caption' key with the sentence to be scrambled.
    - level (float): The level of scrambling to apply. Higher values result in
      more scrambled sentences. -1 indicates filtering at word level (for each
      sentence lowest possible level is selected for scrambling). Values between
      zero and one indicate the fraction of the depth of the sentence structure
      at which to sample.
    - rng (optional): The random number generator used in scrambling. Defaults to None.

    Returns:
    - list of dict: A new dataset with scrambled sentences. Each element in the list is a dictionary similar
      to the input dataset but with the 'caption' key containing the scrambled
      sentence and the tokens_positive_eval containing the locations of the
      scrambled phrases.
    """
    if not rng:
        rng = np.random.default_rng()
    dataset = deepcopy(dataset)
    dataset["images"] = deepcopy(dataset["images"])
    excluded_counter = 0
    for i, image in enumerate(tqdm(dataset["images"])):
        sentence = image["caption"]
        phrase_positions = np.array(image["tokens_positive_eval"]).reshape(-1,2)
        unique_phrases, inverse = np.unique(phrase_positions, axis=0, return_inverse=True)
        scrambled_sentence, scrambled_phrase_positions = scramble(sentence, unique_phrases, level, rng)
        image["caption"] = scrambled_sentence
        image["tokens_positive_eval"] = np.array(scrambled_phrase_positions)[np.array(inverse)].reshape(-1,1,2).tolist()
    print(f"Excluded {excluded_counter} samples.")
    return dataset