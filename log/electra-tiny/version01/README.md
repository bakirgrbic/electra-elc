Pre-training hyperparameters:
- Epochs: 10
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: AdamW
- Compute: AWS Sagemaker ml.g4dn.xlarge

finetuning (`personal_evaluation`) hyperparameters:
- Epochs: 3
- Batch size: 64
- Learning rate: 2e-5
- Optimizer: AdamW
- Max Length: 128
- Parameter Freezing: None
- Compute: AWS Sagemaker ml.g4dn.xlarge

blimp hyperparameters:
- Max epochs: 1
- Script modified for masked LMs.
- Compute: Mac M2
