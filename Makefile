APP_NAME=summarypy  
CONTAINER_NAME=summarypy    
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path)))) 

.PHONY: build
.PHONY: ria_20k
.PHONY: ria_1k
.PHONY: ria_full
.PHONY: clean

build:
	sudo docker build -t ${APP_NAME} -f Dockerfile .
ria_20:
	wget https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria_20.json -P $(PWD)/data/
ria_1k:
	wget https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria_1k.json -P $(PWD)/data/
ria_full:
	wget  https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria.json.gz -P $(PWD)/data/
	gunzip $(PWD)/data/ria.json.gz
clean:
	rm -rf $(PWD)/data/ria.json
	rm -rf $(PWD)/data/ria_20.json
	rm -rf $(PWD)/data/ria_1k.json