# summary-py
## Overview
summary-py is a simple wrapper around a code from repository [PreSumm](https://github.com/nlpyang/PreSumm). 
## Usage
1. Download ria news datasets for **extractive summarization**
```
make ria_20
```
```
make ria_1k
```
```
make ria_full
```
2. Build docker image
```
make build 
```
3. Change data path `config/ext_summary.yaml`

4. Run training
```
docker run --gpus all summarypy
```
## TODO
- [ ] Add abstractive summarization.

## License
Licensed under MIT license.
