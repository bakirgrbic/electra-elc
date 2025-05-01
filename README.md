# cs-nlp-finalproject

## Description
Final project for CS 436 to learn more about pre-training and time 
optimizations to the process.

## How to Run
Tested on MacOS and AWS Sagemaker environments. 
When running on AWS Sagemaker make sure to install the `transformers` and 
`torch` libraries.

### Pretraining
Tested to work with ELECTRA and BERT models.
TODO notebook
TODO scripts

### Evaluation
#### Web of Science Classification
Tested to work with ELECTRA and BERT models.
TODO notebook
TODO script

#### evaluation-pipeline-2024
1. Follow [Install](https://github.com/babylm/evaluation-pipeline-2024/tree/main?tab=readme-ov-file#install) and [Data](https://github.com/babylm/evaluation-pipeline-2024/tree/main?tab=readme-ov-file#data) instructions from the original evaluation repository. If using AWS install all libraries with pip in a terminal.
2. Make sure `evaluation_data` directory is in the root of `evaluation-pipeline-2024` project
3. Move files from `tokenizer` directory to the `checkpoint/MODEL_NAME` for the model that will be evaluated
4. Open a terminal in the the root of `evaluation-pipeline-2024` project.

To Evaluate on BLiMP first modify the `./eval_blimp.sh` script to be the following
```shell
#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

python -m lm_eval --model hf-mlm \
    --model_args pretrained=$MODEL_PATH,backend="mlm" \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json
```
Run the following command to evaluate on BLiMP for given model path such as ELECTRA_PT.
```[shell](shell)
./eval_blimp.sh ../../checkpoints/ELECTRA_PT
```

To Evaluate on (Super)GLUE run the following command:
```shell
./finetune_model.sh PATH_TO_MODEL
```
NOTE: If (Super)GLUE evalution is not working make sure transormers lib is version 4.49.0 

Results for these evaluations will be found in the `evaluation-pipeline-2024/results`


## Acknowledgments
Code from [models/electra_elc.py](./models/electra_elc.py) is a derivative work
of [transformers ELECTRA model](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/electra/modeling_electra.py) 
and [ELC-BERT zero initialization model](https://github.com/ltgoslo/elc-bert/blob/main/models/model_elc_bert_zero.py).
