CONDA_ENV=PytorchLightningEnv
echo "Cleaning up older environment"
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda env remove -n $CONDA_ENV
conda env create -f environment.yml
conda activate $CONDA_ENV
echo "Activated "$CONDA_ENV
echo "Installing spaCy models..."
spacy download en_core_web_md
spacy download en_core_web_sm
spacy download en
# echo "Installing neuralcoref..."
# git clone https://github.com/huggingface/neuralcoref.git ~/neuralcoref
# cd ~/neuralcoref/
# pip install -r requirements.txt
# pip install -e .
# cd -
# echo "Deleting neuralcoref folder..."
# rm -r ./neuralcoref
echo "Removing unused packages..."
conda clean -ay
