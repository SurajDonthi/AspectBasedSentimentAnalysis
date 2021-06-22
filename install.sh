echo "Cleaning up older environment"
conda env remove -n NLPDev
conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate NLPDev
echo "Activated NLPDev"
echo "Installing spaCy models..."
spacy download en_core_web_md
spacy download en_core_web_sm
spacy download en
echo "Installing neuralcoref..."
git clone https://github.com/huggingface/neuralcoref.git ~/neuralcoref
cd ~/neuralcoref/
pip install -r requirements.txt
pip install -e .
cd -
# echo "Deleting neuralcoref folder..."
# rm -r ./neuralcoref
echo "Removing unused packages..."
conda clean -ay
