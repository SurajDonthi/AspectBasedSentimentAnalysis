echo "Cleaning up older environment..."
conda env remove -n NLPDev
echo "Creating environment NLPDev..."
conda env create -f environment.yml
conda activate NLPDev
echo "Activated NLPDev"
echo "Installing spaCy models..."
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
python -m spacy download en
echo "Installing neuralcoref..."
git clone https://github.com/huggingface/neuralcoref.git $HOME/neuralcoref
Push-Location $HOME/neuralcoref/
pip install -r requirements.txt
pip install -e .
Pop-Location
# echo "Deleting neuralcoref folder..."
# rm -r ./neuralcoref
echo "Removing unused packages..."
conda clean -ay
