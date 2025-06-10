SHELL = /bin/zsh
# Below is needed for conda to function in subshells for each command that needs 
# conda
CONDA_INIT = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ;

.PHONY: all-tests
all-tests:
	@$(CONDA_INIT) conda activate bblm ; python3 -m pytest -v tests/

.PHONY: slow-tests
slow-tests:
	@$(CONDA_INIT) conda activate bblm ; python3 -m pytest -v -m "slow" tests/

.PHONY: fast-tests
fast-tests:
	@$(CONDA_INIT) conda activate bblm ; python3 -m pytest -v -m "not slow" tests/

.PHONY: env
env:
	@$(CONDA_INIT) conda env create -f environment.yml
