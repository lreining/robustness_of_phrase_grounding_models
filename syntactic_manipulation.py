import itertools
from tqdm import tqdm
import numpy as np

from dataset_utils.syntactic_tree import SyntacticTree
from torch_geometric.transforms import BaseTransform
MASK = "XXXX"

def scramble_sentence(sentence, level,seed=1):
    tree = SyntacticTree(sentence)
    subsentences, _ = tree.get_sentence_parts_for_level(level)
    np.random.seed(seed)
    np.random.shuffle(subsentences)
    return subsentences


def scramble_phrase(sentence,level, phrase_idx_bounds, seed=1):
    tree = SyntacticTree(sentence)
    sub_sentences,_ = tree.get_sentence_parts_for_level(level)
    sub_phrases = []
    idx = 0
    for sub_sentence in sub_sentences:
        if idx >= phrase_idx_bounds[0] and idx < phrase_idx_bounds[1]:
            sub_phrases.append(sub_sentence)
        idx += len(str(sub_sentence))+1
    np.random.seed(seed)
    np.random.shuffle(sub_phrases)
    return sub_phrases

def scramble_with_multiple_phrases(sentence, phrase_idx_bounds, level, seed=1, is_phrase_scrambled=False):
    masked_sentence = sentence
    decrement = 0
    for i,idx in enumerate(phrase_idx_bounds):
        masked_sentence = mask_phrase(masked_sentence, [idx[0] - decrement, idx[1] - decrement], str(i))
        decrement += idx[1]-idx[0]-len(MASK+str(i))
    sub_sentences = scramble_sentence(masked_sentence, level, seed)
    sub_phrases = []
    if is_phrase_scrambled:
        for idx in phrase_idx_bounds:
            sub_phrases.append(join_sentence(scramble_phrase(sentence, level, idx, seed)))
    else:
        sub_phrases = []

    if len(sub_phrases) == 0:
        for idx in phrase_idx_bounds:
            sub_phrases.append(sentence[idx[0]:idx[1]])
    sub_sentences = join_sentence(sub_sentences)
    new_phrase_idx_bounds = []
    for i, sub_phrase in enumerate(sub_phrases):
        sub_sentences, b = unmask_phrase(sub_sentences, sub_phrase, str(i))
        new_phrase_idx_bounds.append(b)
    return sub_sentences.lower(), new_phrase_idx_bounds

def scramble(sentence, phrase_idx_bounds,level, seed=1, is_phrase_scrambled=False):
    masked_sentence = mask_phrase(sentence, phrase_idx_bounds)
    sub_sentences = scramble_sentence(masked_sentence, level, seed)
    if is_phrase_scrambled:
        sub_phrases = scramble_phrase(sentence, level, phrase_idx_bounds, seed)
    else:
        sub_phrases = []
    sub_sentences, sub_phrases = join_sentence(sub_sentences), join_sentence(sub_phrases)
    if len(sub_phrases) == 0:
        sub_phrases = sentence[phrase_idx_bounds[0]:phrase_idx_bounds[1]]
    s, b = unmask_phrase(sub_sentences, sub_phrases)
    return s.lower(), b

def mask_phrase(sentence, phrase_idx_bounds,add=""):
    start_idx, end_idx = phrase_idx_bounds
    sentence = sentence[:start_idx]+MASK+add+sentence[end_idx:]
    return sentence

def join_sentence(subsentences):
    subsentences = list(map(str, subsentences))
    return " ".join(subsentences)

def unmask_phrase(sentence, phrase,add=""):
    idx = sentence.find(MASK+add)
    sentence = sentence[:idx] + phrase + sentence[idx+len(MASK+add):]
    return sentence, [idx, idx+len(phrase+add)]

class ScrambledSentence(BaseTransform):
    def __init__(self, level, seed=1):
        """
        Initializes a ScrambledSentence object to scramble the captions
        within the dataset.

        Parameters:
        - level (int): The level of scrambling for the sentence.
        - seed (int): The seed value for randomization (default is 1).
        """
        self.level = level
        self.seed = seed
    
    def __call__(self, data):
        for sample in tqdm(data):
            sentence = sample[1]["caption"]
            scrambled_sentence, scrambled_phrase_positions = scramble_with_multiple_phrases(sentence, np.array(sample[1]["tokens_positive"]).reshape(-1,2), self.level, self.seed)
            sample[1]["scrambled_caption"] = scrambled_sentence
            sample[1]["scrambled_tokens_positive"] = scrambled_phrase_positions
        return data