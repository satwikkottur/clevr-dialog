# CLEVR-Dialog

This repository contains code for the paper:

**CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog**  
[Satwik Kottur][2], [José M. F. Moura][3], [Devi Parikh][4], [Dhruv Batra][5], [Marcus Rohrbach][6]  
[[PDF][7]] [[ArXiv][1]] [[Code][8]]  
_Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2019_

### Abstract
Visual Dialog is a multimodal task of answering a sequence of questions grounded in an image, 
using the conversation history as context.
It entails challenges in vision, language, reasoning, and grounding.
However, studying these subtasks in isolation on large, real datasets is infeasible as it
requires prohibitively-expensive complete annotation of the 'state' of all images and dialogs. 


We develop CLEVR-Dialog, a large diagnostic dataset for studying multi-round reasoning in 
visual dialog.
Specifically, we construct a dialog grammar that is grounded in the scene graphs of the 
images from the CLEVR dataset.
This combination results in a dataset where all aspects of the visual dialog are fully
annotated.
In total, CLEVR-Dialog contains 5 instances of 10-round dialogs for about 85k CLEVR images,
totaling to 4.25M question-answer pairs. 


We use CLEVR-Dialog to benchmark performance of standard visual dialog models; in
particular, on visual coreference resolution (as a function of the coreference distance).
This is the first analysis of its kind for visual dialog models that was not possible
without this dataset.
We hope the findings from CLEVR-Dialog will help inform the development of future models
for visual dialog.


[![CorefNMN](https://i.imgur.com/mwC1mVC.png)][1]
This repository generates a version of our diagnostic dataset **CLEVR-Dialog**.
(figure above).


If you find this code useful, consider citing our work:

```
@inproceedings{Kottur2019CLEVRDialog,
	title={CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog},
	author={Satwik Kottur and Jos'e M. F. Moura and Devi Parikh and Dhruv Batra and Marcus Rohrbach},
	year={2019}
}
```


## Setup

The code is in Python3 with following python package dependencies:

```bash
pip install absl-py
pip install json
pip install tqdm
pip install numpy
```

### Directory Structure
The repository contains the following files:


* `generate_dataset.py`: Main script to generate the dataset
* `constraints.py`: List of constraints for caption and question generation
* `clevr_utils.py`: Utility functions to dialog generation
* `global_vars.py`: List of global variables along with initialization

In addition, the dataset generation code requires following files:

* `templates/synonyms.json`: Compilation of words and their synonyms
* `templates/metainfo.json`: Contains information about attributes and their values for CLEVR objects
* `templates/captions` and `templates/questions`: Caption and question templates respectively.

## Generating CLEVR-Dialog Dataset

To generate the dataset, please check `run_me.sh`.

The dataset used in the paper will be released soon!

## Contributors

* [Satwik Kottur][2]

## Contributors

This project is licensed under the license found in the LICENSE file in the root directory of this source tree ([here][9]).


[1]:https://arxiv.org/abs/1903.03166
[2]:https://satwikkottur.github.io/
[3]:https://users.ece.cmu.edu/~moura/
[4]:https://www.cc.gatech.edu/~parikh/
[5]:https://www.cc.gatech.edu/~dbatra/
[6]:http://rohrbach.vision/
[7]:https://arxiv.org/abs/1903.03166.pdf
[8]:https://github.com/satwikkottur/clevr-dialog
[9]:https://github.com/satwikkottur/clevr-dialog/blob/master/LICENSE