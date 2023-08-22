# Stochastic Local Search Algorithm with Heuristics based on Machine Learning


This Project implements the SLS WalkAAF algorithm and adds Heuristics based on Machine Learning and other Heuristics to find a single stable extension of an Abstract Argumentation Framework (SE-ST).
This Codebase was done for my [Bachelor's Thesis](https://github.com/kon-drees/sls-ml-thesis).



## Overview

This Python Project implements a Stochastic Local Search Algorithm with Heuristics based on Machine Learnig for findind a singe stable extension. 
The Machine Learning Models are trained with sklearn and PyTorch and used for the Heuristcs. There exist two models. One for finding the next Argument for switching its current Labelling and one for the initial Labelling of the Arguments in an Abstract Argumentation Framework





## Requirements

The [Light GBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) classifier is installed.
[TorchGeometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) is installed.
Libraries under requirements.txt are installed.




## Usage 


The Python file af_generator.py can generate argumentation frameworks using jAFBenchGen. To generate abstract argumentation frameworks, you need the jAFBenchGen jar file location and set it in the Python file under:

```python
# location of the jar file
command = f"java -jar /location/of/the/jar/jAFBenchGen.jar -numargs {num_args} -type {af_type} {additional_parameters}"
```

Also, set the output folder for the generated frameworks:

```python
output_folder = '/location/of/the/generated_argumentation_frameworks'
```
Run the code with the Main method

For feature extractions of the argumentation framework and data labeling, use the main methods in af_feature_extractor.py and af_ml_label_calculator.py Python files.

Set the locations of the output data and argumentation frameworks in the Main methods of each Python files like this:

```python
if __name__ == '__main__':
    # paths
    arg_folder = '/location/of/folder/sls-ml/files/argumentation_frameworks'
    output_folder = '/location/of/folder/sls-ml/files/processed_argumentation_frameworks'
    processed_files = '/location/of/folder/sls-ml/files/processed_files.txt'
    preprocess_data(arg_folder, output_folder, processed_files)

```

The Python file af_ml_model_creator.py and af_nn_model_creator.py create the models using classical classifiers like Random Forests and Neural Networks like Graph Convolutional Networks.

Set the location of the file paths in the main methods.


To use the different SLS-Algorithm use the functions from the walkaaf.py packages:

For example use walkaaf_with_ml3_nn, which uses Neural Networks for finding single extensions:


```python
walkaaf_with_ml3_nn(af_graph, model_rn_red, model_in_red)
```

The function needs the loaded models and an abstract argumentation framework and returns a single stable extension if found.

















