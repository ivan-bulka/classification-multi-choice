# Classification Multi choice 

----
## Installing:
1. Install poetry: 

`pip install poetry` 

2. Install packages with poetry:

`poetry install`

## Run experiments 
1. Run jupyter notebook in the root:
`jupyter notebook`
2. Open a file `Classification of Multiple-Choice Questions.ipynb` where is all important information

## Blocks modules description
1. `subjects-questions.csv` - a dataset for training. 
2. `text_processing.py` - a collection of methods for text preprocessing
3. `encoding_data.py` - embedding data in batches
4. `text_classifier.py` - classification for embedded data with XGBoost
5. `chat_gpt_classifier.py` - GPT classifier with Open AI GPT-4 model and prompt engineering

