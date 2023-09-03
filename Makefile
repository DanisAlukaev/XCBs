SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

set-up-env:
	conda create --name autoconcept --file conda-linux-64.lock; \
	$(CONDA_ACTIVATE) autoconcept; \
	poetry install; \

update-env:
	conda-lock -k explicit --conda mamba; \
	mamba update --file conda-linux-64.lock; \

download_data: create_data_dir download_shapes download_coco download_cub download_mimic

create_data_dir:
	cd autoconcept/; \
	mkdir -p data/; \

download_shapes:
	cd autoconcept/data/; \
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WBjt4WDt5eIVlEAf9SYH-9j-UiIAyF9Q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WBjt4WDt5eIVlEAf9SYH-9j-UiIAyF9Q" -O shapes.zip && rm -rf /tmp/cookies.txt; \
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n2UjID5uv_3_lDQMVA3Ioga5RHBzMICS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n2UjID5uv_3_lDQMVA3Ioga5RHBzMICS" -O shapes-hard.zip && rm -rf /tmp/cookies.txt; \
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HPAOqLvU7V7bGS291D0H-qLnWiqUMStg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HPAOqLvU7V7bGS291D0H-qLnWiqUMStg" -O shapes-hard-2.zip && rm -rf /tmp/cookies.txt; \
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MujTdaD6F7ON-j3nkBM7h-9T2y3UO9YT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MujTdaD6F7ON-j3nkBM7h-9T2y3UO9YT" -O shapes-hard-3.zip && rm -rf /tmp/cookies.txt; \
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g20ZRd9uJ9B6FOZxw-dLyaJ6Mn5yhERS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g20ZRd9uJ9B6FOZxw-dLyaJ6Mn5yhERS" -O shapes-hard-4.zip && rm -rf /tmp/cookies.txt; \
	unzip shapes.zip; \
	unzip shapes-hard.zip; \
	unzip shapes-hard-2.zip; \
	unzip shapes-hard-3.zip; \
	unzip shapes-hard-4.zip; \
	rm *.zip; \

download_coco:
	cd autoconcept/data/; \
	wget -c --read-timeout=5 --tries=0 http://images.cocodataset.org/zips/train2017.zip; \
	unzip train2017.zip; \
	wget -c --read-timeout=5 --tries=0 http://images.cocodataset.org/annotations/annotations_trainval2017.zip; \
	unzip annotations_trainval2017.zip; \
	rm *.zip; \

download_cub:
	cd autoconcept/data/; \
	wget -c --read-timeout=5 --tries=0 https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz; \
	tar -xf CUB_200_2011.tgz; \
	rm *.tgz; \

download_mimic:
	cd autoconcept/data/; \
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
