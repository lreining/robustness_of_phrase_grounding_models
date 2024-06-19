#%%
from syntactic_manipulation import scramble
from nlp_utils.syntactic_tree import SyntacticTree

#%%
sentence = "Several climbers in a row are climbing the rock."
phrase_idx_bounds = [39,47]
print("Sentence:")
print(sentence)
print("Phrase:")
print(f"{sentence[phrase_idx_bounds[0]:phrase_idx_bounds[1]+1]} ({tuple(phrase_idx_bounds)})")

#%%
tree = SyntacticTree(sentence)
tree.show()
#%%
level  = -1
scrambled_sentence, scrambled_phrase_idx_bounds = scramble(sentence,level, phrase_idx_bounds)
print("Scrambled Sentence:")
print(scrambled_sentence)
print("Scrambled Phrase:")
print(f"{scrambled_sentence[scrambled_phrase_idx_bounds[0]:scrambled_phrase_idx_bounds[1]+1]} ({tuple(scrambled_phrase_idx_bounds)})")