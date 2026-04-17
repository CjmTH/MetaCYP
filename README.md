# MetaCYP
MetaCYP, a multimodal deep learning framework for the joint prediction of bonds of metabolism (BoMs) and reaction types in CYP-mediated metabolic processes. 


<img width="863" height="617" alt="image" src="https://github.com/user-attachments/assets/1130ec97-d08c-4998-b28a-40ed8fe4ef6a" />

## Setup
The environment for CYPMol is configured identically to DeepP450. For detailed setup instructions, please refer to the DeepP450 repository: https://github.com/CjmTH/DeepP450.

## training 
The raw data and training scripts for the BoM and ReactType prediction tasks are available in the main/Data/ and main/Model/ directories, respectively.

## testing
The input file formats required for CYPMol across different tasks can be found in the corresponding BoM_example.csv or ReactType_example.csv within Main/Data/ directory. The model weightare available at: https://huggingface.co/CJM1111/MetaCYP/tree/main.
