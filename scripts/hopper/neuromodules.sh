module load gnu10
module load use.own pyenv

PYTHON_LATEST="$(pyenv latest 3)"
pyenv global $PYTHON_LATEST

python --version
pip --version

source ~/neuromorphic/framework/pyframework/bin/activate
cd ~/neuromorphic

