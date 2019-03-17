# CLEVR-Dialog

This repository contains code for the paper:

**CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog**  
[Satwik Kottur][2], [José M. F. Moura][3], [Devi Parikh][4], [Dhruv Batra][5], [Marcus Rohrbach][6]  
[[PDF][7]] [[ArXiv][1]] [[Code][8]]  
**Oral Presentation**   
_Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2019_

If you find this code useful, consider citing our work:

```
@inproceedings{Kottur2019CLEVRDialog,
	title  = {CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog},  
	author = {Kottur, Satwik and Moura, Jos\'e M. F. and Parikh, Devi and   
	          Batra, Dhruv and Rohrbach, Marcus},  
	journal = {arXiv preprint arXiv:1903.03166},
	year   = {2019}  
}
```

### Abstract
Visual Dialog is a multimodal task of answering a sequence of questions 
grounded in an image, using the conversation history as context.
It entails challenges in vision, language, reasoning, and grounding.
However, studying these subtasks in isolation on large, real datasets is 
infeasible as it requires prohibitively-expensive complete annotation of the 
'state' of all images and dialogs. 


We develop CLEVR-Dialog, a large diagnostic dataset for studying multi-round 
reasoning in visual dialog.
Specifically, we construct a dialog grammar that is grounded in the scene 
graphs of the images from the CLEVR dataset.
This combination results in a dataset where all aspects of the visual dialog 
are fully annotated.
In total, CLEVR-Dialog contains 5 instances of 10-round dialogs for about 85k 
CLEVR images, totaling to 4.25M question-answer pairs. 


We use CLEVR-Dialog to benchmark performance of standard visual dialog models;
in particular, on visual coreference resolution (as a function of the 
coreference distance).
This is the first analysis of its kind for visual dialog models that was not 
possible without this dataset.
We hope the findings from CLEVR-Dialog will help inform the development of 
future models for visual dialog.


[![CorefNMN](https://i.imgur.com/mwC1mVC.png)][1]
This repository generates a version of our diagnostic dataset **CLEVR-Dialog**
(figure above).


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

### CLEVR Images
Our dataset is built on [CLEVR][10] images, which can be downloaded from [here][11].
Extract the images and scene JSON files in `data/` folder.
We will only use CLEVR `train` and `val` splits as scene JSON files are unavailable for `test` split.


## Generating CLEVR-Dialog Dataset

To generate the dataset, please check `run_me.sh`.
Additional details about the supported flags can be found in `generate_dataset.py`.
An example command is shown below:

```bash
DATA_ROOT='data/CLEVR_v1.0/'
python -u generate_dataset.py \
	--scene_path=${DATA_ROOT}"scenes/CLEVR_train_scenes.json" \
	--num_beams=100 \
	--num_workers=1 \
	--save_path=${DATA_ROOT}"clevr_dialog_train_raw.json" \
	--num_images=10
```


## CLEVR-Dialog Annotations

The generated JSON contains a list of dialogs on CLEVR images with following fields:  

* `split`: Specifies if the CLEVR split is train/val. 
* `image_index`: CLEVR image index. 
* `image_filename`: CLEVR image filename. 
* `dialogs`: List of dialog instances for a given image, each with following fields:  
	 &nbsp;&nbsp; |--`caption`: Caption for the dialog instance  
 	 &nbsp;&nbsp; |--`template_info`: Template information for the dialog (caption + 10 questions)  
	 &nbsp;&nbsp; |--`dialog`: Text for the ten rounds of dialog, each with following fields:  
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
	 |--`question`: Question text for the current round  
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
	 |--`answer`: Answer text for the current round  
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
	 |--`template`: Question template for the current round  
 	 &nbsp;&nbsp; |--`graph`: Scene graph information for the dialog, with following fields:  
 	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
	 |--`objects`: Objects with attributes discussed in the dialog  
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
	 |--`counts`: Specific object counts discussed in the dialog  
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
	 |--`relationships`: Object relationships discussed in the dialog  
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
	 |--`exists`: Object existences discussed in the dialog  
	 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
	 |--`history`: List of incremental scene graph information conveyed in each round	
	 
	 
The dataset used in the paper can be downloaded here: [train][13] and [val][12] splits.

## Contributors

* [Satwik Kottur][2]

For any questions, please feel free to contact the above contributor(s).

## License

This project is licensed under the license found in the LICENSE file in the
root directory of this source tree ([here][9]).


[1]:https://arxiv.org/abs/1903.03166
[2]:https://satwikkottur.github.io/
[3]:https://users.ece.cmu.edu/~moura/
[4]:https://www.cc.gatech.edu/~parikh/
[5]:https://www.cc.gatech.edu/~dbatra/
[6]:http://rohrbach.vision/
[7]:https://arxiv.org/abs/1903.03166.pdf
[8]:https://github.com/satwikkottur/clevr-dialog
[9]:https://github.com/satwikkottur/clevr-dialog/blob/master/LICENSE
[10]:https://cs.stanford.edu/people/jcjohns/clevr/
[11]:https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip
[12]:https://drive.google.com/file/d/1efCk917eT_vgDO__OS6cKZkC8stT5OL7/view?usp=sharing
[13]:https://drive.google.com/file/d/1u6YABdNSfBrnV7CXVp5cfOSstGK0gyaE/view?usp=sharing