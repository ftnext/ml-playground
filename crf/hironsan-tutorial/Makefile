help:
	@echo 'Run the following to reproduce:'
	@echo '1. Create venv, then pip install requirements.lock'
	@echo '2. make setup'
	@echo '3. make train'
	@echo '4. make evaluate'

setup:
	@mkdir -p data
	@wget https://raw.githubusercontent.com/Hironsan/IOB2Corpus/master/hironsan.txt -O data/hironsan.txt

train:
	@python train.py

evaluate:
	@python evaluate.py
