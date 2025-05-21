SHELL = /bin/zsh
# Below is needed for conda to function in subshells for each command that needs 
# conda
CONDA_INIT = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ;

.PHONY: test
test:
	@$(CONDA_INIT) conda activate bblm ; python3 -m unittest tests/*.py

.PHONY: env
env:
	@$(CONDA_INIT) conda env create -f environment.yml
