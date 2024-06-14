import itertools

import numpy as np

from nlp_utils.syntactic_tree import SyntacticTree

MASK = "XXXX"
def get_sentence_permutations(sentence, level):
    tree = SyntacticTree(sentence)
    subsentences,_ = tree.get_sentence_parts_for_level(level)
    permutations = list(itertools.permutations(subsentences))
    permutations.remove(tuple(subsentences))
    return permutations

def scramble_sentence(sentence, level,seed=1):
    tree = SyntacticTree(sentence)
    subsentences, _ = tree.get_sentence_parts_for_level(level)
    np.random.seed(seed)
    np.random.shuffle(subsentences)
    return subsentences
def mask_phrase(sentence, phrase_config):
    phrase_idx, phrase = phrase_config["first_word_index"], phrase_config["phrase"]
    sentence = sentence[:phrase_idx]+MASK+sentence[phrase_idx+len(phrase):]
    return sentence

def unmask_phrase_from_scrambled_sentence(subsentences, phrase_config):
    phrase_config = phrase_config.copy()
    phrase = phrase_config["phrase"]
    #TODO: scramble phrase if on specified level (but only scramble parts within the phrase)
    subsentences = list(map(str, subsentences))
    sentence = " ".join(subsentences)
    idx = sentence.find(MASK)
    phrase_config["first_word_index"] = idx
    sentence = sentence[:idx] + phrase + sentence[idx+len(MASK):]
    return sentence, phrase_config

def get_permutations_for_image_phrase(sentence, phrase_config, level):
    sentence = mask_phrase(sentence, phrase_config)
    permutations = get_sentence_permutations(sentence, level)
    sentences = [unmask_phrase_from_scrambled_sentence(subsentences, phrase_config) for subsentences in permutations]
    return sentences

def scramble_sentence_for_image_phrase(sentence, phrase_config, level,seed=1):
    sentence = mask_phrase(sentence, phrase_config)
    subsentences = scramble_sentence(sentence,level,seed)
    sentence, phrase_config = unmask_phrase_from_scrambled_sentence(subsentences, phrase_config)
    return sentence, phrase_config