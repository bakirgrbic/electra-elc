Pre-training hyperparameters:
- Epochs: 9 done but 10 in total (picked up pretraining from version01)
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: AdamW
- Compute: AWS Sagemaker ml.g4dn.xlarge

blimp hyperparameters:
- Max epochs: 1
- Script modified for masked LMs.
- Compute: Mac M2
