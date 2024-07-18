# Robustness of State-of-the-Art Visual Grounding Models using Corrupted Linguistic Dependency Structures

## Motivation

A key challenge in machine learning research is the lack of resilience to corruption in existing unimodal models.
Despite the advantages of using multiple modalities, the issue of corruption robustness persists, with limited prior research focusing
on out-of-domain or corruption robustness in the multimodal setting, particularly in phrase grounding. This raises concerns
about the suitability of such models for safety-critical applications.
Here, we evaluate the robustness of state-of-the-art phrase grounding models using corrupted linguistic dependency
structures. This form of corruption allows us to assess the influence of structure on model performance. By analyzing models of
varying sizes and architectures on our corrupted phrase grounding dataset, we aim to explore the interaction between robustness and
accuracy in different models. We hypothesize that the accuracy of grounding models tends to decrease as the textual prompt contains
less syntactic structure.

## Repository Structure

### Scripts

- `create_test_dataset.py`: Creates a test dataset with different levels of
  syntactic scrambling based on the flickr30k dataset.
- `plot.py`: Analyze and plot the results.
- `create_analysis_plots_sentence_level.ipynb`: Analyze the influence of sentence structure on the performance of the models.

### Directories:

- `./dataset_utils`: Utils for working with the flickr30k dataset.
- `./flicker_test_dataset`: Contains annotations which are scrambled at different
  syntactic depths.
- .`/flickr_test_datasets_sentence_metrics`: Contains different evaluation results
  to avoid having to recompute them.
- `./figures`: Contains the figures produced in the analysis.

## Dataset

The dataset is based on the flickr30k dataset. We provide the original dataset
and the corrupted dataset in the `flicker_test_dataset` directory. The dataset
was created by scrambling the dependency structure of the sentences in the
flickr30k dataset. Here, we provide scrambled versions of the test dataset.
Scrambling was performed using the `create_test_dataset.py` script.

### Manipulations

We provide an open-source method for corrupting sentences of the flickr30k entities dataset with systematic control of the degree to which
the original sentence structure is preserved.

To perform the scrambling:
- for single sentence: `dataset_utils.syntactic_manipulation.scramble`
- for whole dataset: `dataset_utils.syntactic_manipulation.scramble_dataset`

For example on usage, refer to `create_test_dataset.py`.


Our approach to scramble sentence s at level l:
0. Sentence is converted to lowercase and punctuation is removed
1. The target phrases - given by their index bounds `phrase_idx_bounds` - are masked so that their structure will be preserved (if such behavior is not desired: `is_phrase_scrambled = True`)
2. Construct the syntactic tree (`dataset_utils.syntactic_tree.SyntacticTree`) of the sentence that is to be scrambled
3. Retrieve constituent parts at level min(l, i) for each branch of the syntactic tree where l refers to the scrambling level and i to the maximum height of the current branch
4. Shuffle original sentence s based on constituent parts


## Model Setups

### GLIP
We used the models provided on the following GitHub page: [https://github.com/microsoft/GLIP]. Specifically we used GLIP-T (C), GLIP-T and GLIP-L. The code was published under MIT license.

To test the models, complete the following steps:
0. If you haven't downloaded the models yet run the run_models.sh file
1. To start the docker image run the `run_docker.sh` file. Adjust the mounts as needed
2. Within the docker image, run the `run_models.sh` file. Adjust the output directory

### Fiber
Just like GLIP, we used a model provided by the following GitHub page: [https://github.com/microsoft/FIBER]. 

To test the model, complete the following steps:
0. Navigate to the FIBER folder
1. Run the `docker_run.sh` file to pull and run the necessary docker environment for running the model
   1. Make sure to adjust mounting directories accordingly   
2. Navigate to the actual model (folder: `/FIBER/FIBER`)
3. Run the `setup_and_run.sh` file
   1. Also adjust paths as needed. For more information, refer to [https://github.com/microsoft/FIBER/blob/main/fine_grained/README.md]

### MDETR
The MDETR code was provided by Facebook Research and can be found on this GitHub page: [https://github.com/facebookresearch/multimodal/tree/main]. The code is slightly adjusted to use for our purposes and make it more concise. The mdetr code was published under a BSD-3-Clause license.

To test the model, complete the following steps:
0. Adjust the parameters in the `phrase_grounding.json` file located in `./mdetr `
  0. `device`: device to compute the metrics on
  1. `flickr_img_path`: path of the images
  2. `flickr_dataset_path`: path of the whole dataset folder
  3. `flickr_ann_path`: path of the annotations files
  4. The other parameter can also be adjusted, but are optional
1. Navugate to the mdetr folder
2. Run `python phrase_grounding.py` 
3. After running the experiments, the results can be found in `./mdetr/results`  

