download_data: create_data_dir download_coco download_cub download_mimic

create_data_dir:
	mkdir -p data/; \

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

preprocess_data:
	cd autoconcept/; \
	python -m scripts.majority_voting; \
	python -m scripts.merge_datasets; \

download_mimic:
	cd data/; \
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1W4fgV85QJBkJV-6dWaF3DpdgwW_0h8Y0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1W4fgV85QJBkJV-6dWaF3DpdgwW_0h8Y0" -O mimic-cxr.zip && rm -rf /tmp/cookies.txt; \
	unzip mimic-cxr.zip; \
	rm *.zip; \
