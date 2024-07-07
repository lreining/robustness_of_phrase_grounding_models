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

- `./GLIP`: Contains the code for the GLIP models.
- `./Fiber`: Contains the code for the Fiber model.
- `./MDETR`: Contains the code for the MDETR model.
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

## Model Setups

### GLIP
We used the models provided on the following github page: [https://github.com/microsoft/GLIP]. Specifically we used GLIP-T (C), GLIP-T and GLIP-L.

To test the models, complete the following steps:
0. If you haven't downloaded the models yet run the run_models.sh file
1. To start the docker image run the run_docker.sh file. Adjust the mounts as needed
2. Within the docker image, run the run_models.sh file. Adjust the output directory

### Fiber

### MDETR

## Dataset and variations