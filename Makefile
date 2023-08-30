download_data: create_data_dir download_shapes download_coco download_cub download_mimic

SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

first-time-set-up-env:
	wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh; \
	bash Mambaforge-Linux-x86_64.sh; \
	conda env create -f environment.yml; \
	$(CONDA_ACTIVATE) autoconcept; \
	conda-lock -k explicit --conda mamba; \
	poetry init --python=~3.10; \
	poetry add --lock torch=1.12.1 torchaudio=0.12.1 torchvision=0.13.1; \
	poetry add --lock conda-lock; \

set-up-env:
	conda create --name autoconcept --file conda-linux-64.lock; \
	$(CONDA_ACTIVATE) autoconcept; \
	poetry install; \

update-env:
	conda-lock -k explicit --conda mamba; \
	mamba update --file conda-linux-64.lock; \
	poetry update; \

create_data_dir:
	mkdir -p data/; \

download_shapes:
	cd data/; \
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rHhva_-GS-xUOomhIgCPgnE0lrvh9-eL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rHhva_-GS-xUOomhIgCPgnE0lrvh9-eL" -O shapes.zip && rm -rf /tmp/cookies.txt; \
	unzip shapes.zip; \
	rm *.zip; \

download_coco:
	cd data/; \
	wget -c --read-timeout=5 --tries=0 http://images.cocodataset.org/zips/train2017.zip; \
	unzip train2017.zip; \
	wget -c --read-timeout=5 --tries=0 http://images.cocodataset.org/annotations/annotations_trainval2017.zip; \
	unzip annotations_trainval2017.zip; \
	rm *.zip; \

download_cub:
	cd data/; \
	wget -c --read-timeout=5 --tries=0 https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz; \
	tar -xf CUB_200_2011.tgz; \
	rm *.tgz; \

download_mimic:
	cd data/; \
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1W4fgV85QJBkJV-6dWaF3DpdgwW_0h8Y0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1W4fgV85QJBkJV-6dWaF3DpdgwW_0h8Y0" -O mimic-cxr.zip && rm -rf /tmp/cookies.txt; \
	unzip mimic-cxr.zip; \
	rm *.zip; \

preprocess_cub:
	cd autoconcept/; \
	python -m scripts.majority_voting; \
	python -m scripts.merge_datasets; \

preprocess_mimic:
	cd autoconcept/; \
	python -m scripts.preprocess_mimic_text; \
