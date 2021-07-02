$CONDA_ENV="PytorchLightningEnv"
echo "Cleaning up older environment..."
conda env remove -n $CONDA_ENV
echo "Creating environment $CONDA_ENV..."
conda env create -f environment.yml
conda activate $CONDA_ENV
echo "Activated $CONDA_ENV"
echo "Installing spaCy models..."
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
python -m spacy download en
# echo "Installing neuralcoref..."
# git clone https://github.com/huggingface/neuralcoref.git $HOME/neuralcoref
# Push-Location $HOME/neuralcoref/
# pip install -r requirements.txt
# pip install -e .
# Pop-Location
# echo "Deleting neuralcoref folder..."
# rm -r ./neuralcoref
echo "Removing unused packages..."
conda clean -ay
