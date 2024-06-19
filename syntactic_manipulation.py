import itertools

import numpy as np

from nlp_utils.syntactic_tree import SyntacticTree

MASK = "XXXX"
def get_sentence_permutations(sentence, level):
    #TODO: include scrambling of phrase
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

def scramble_phrase(sentence,level, phrase_idx_bounds, seed=1):
    tree = SyntacticTree(sentence)
    sub_sentences,_ = tree.get_sentence_parts_for_level(level)
    sub_phrases = []
    idx = 0
    for sub_sentence in sub_sentences:
        if idx >= phrase_idx_bounds[0] and idx <= phrase_idx_bounds[1]:
            sub_phrases.append(sub_sentence)
        idx += len(str(sub_sentence))+1
    np.random.seed(seed)
    np.random.shuffle(sub_phrases)
    return sub_phrases

def scramble(sentence, level, phrase_idx_bounds, seed=1, is_phrase_scrambled=False):
    masked_sentence = mask_phrase(sentence, phrase_idx_bounds)
    sub_sentences = scramble_sentence(masked_sentence, level, seed)
    if is_phrase_scrambled:
        sub_phrases = scramble_phrase(sentence, level, phrase_idx_bounds, seed)
    else:
        sub_phrases = []
    sub_sentences, sub_phrases = join_sentence(sub_sentences), join_sentence(sub_phrases)
    if len(sub_phrases) == 0:
        sub_phrases = sentence[phrase_idx_bounds[0]:phrase_idx_bounds[1]]
    return unmask_phrase(sub_sentences, sub_phrases)

def mask_phrase(sentence, phrase_idx_bounds):
    start_idx, end_idx = phrase_idx_bounds
    sentence = sentence[:start_idx]+MASK+sentence[end_idx:]
    return sentence

def join_sentence(subsentences):
    subsentences = list(map(str, subsentences))
    return " ".join(subsentences)

def unmask_phrase(sentence, phrase):
    idx = sentence.find(MASK)
    sentence = sentence[:idx] + phrase + sentence[idx+len(MASK):]
    return sentence.lower(), [idx, idx+len(phrase)]

def scramble_sentence_for_flickr_phrase_annotation(sentence, phrase_config, level,seed=1):
    idx = phrase_config["first_word_index"]
    phrase = phrase_config["phrase"]
    sentence, phrase_idx_bounds = scramble_phrase(sentence, level, [idx, idx+len(phrase)-1], seed)
    phrase_config["phrase"] = sentence[phrase_idx_bounds[0]:phrase_idx_bounds[1]+1]
    phrase_config["first_word_index"] = phrase_idx_bounds[0]
    return sentence, phrase_config