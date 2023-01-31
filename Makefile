download_data: create_data_dir download_coco download_cub

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
