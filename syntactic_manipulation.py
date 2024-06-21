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
        idx += len(sub_sentence)+1
    np.random.seed(seed)
    np.random.shuffle(sub_phrases)
    return sub_phrases

def scramble(sentence, phrase_idx_bounds, level, seed=1, is_phrase_scrambled=False):
    phrase_idx_bounds = phrase_idx_bounds[phrase_idx_bounds[:,0].argsort()]
    masked_sentence = sentence
    decrement = 0
    for i,idx in enumerate(phrase_idx_bounds):
        mask = "".join([MASK,str(i)])
        masked_sentence = mask_phrase(masked_sentence, [idx[0] - decrement, idx[1] - decrement], mask)
        decrement += idx[1]-idx[0]-len(mask)
    sub_sentences = scramble_sentence(masked_sentence, level, seed)
    if is_phrase_scrambled:
        sub_phrases = []
        for idx in phrase_idx_bounds:
            sub_phrase = " ".join(scramble_phrase(sentence, level, idx, seed))
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
            phrase_positions = np.array(sample[1]["tokens_positive"]).reshape(-1,2)
            unique_phrases, inverse = np.unique(phrase_positions, axis=0, return_inverse=True)
            scrambled_sentence, scrambled_phrase_positions = scramble(sentence, unique_phrases, self.level, self.seed)
            sample[1]["scrambled_caption"] = scrambled_sentence
            sample[1]["scrambled_tokens_positive"] =np.array(scrambled_phrase_positions)[np.array(inverse)]
        return data