# StreamETM

StreamETM is an application designed for dynamic topic modeling using the Embedded Topic Model (ETM). It processes streaming text data, merges topic models over time, and detects change points in topic distributions.

## Features

- **Dynamic Topic Modeling**: Continuously update topic models with new data chunks.
- **Topic Merging**: Merge new topic models with existing ones to maintain a coherent topic structure.
- **Change Point Detection**: Detect significant changes in topic distributions over time using Online Change Point Detection (OCPD).
- **Preprocessing**: Preprocess text data including lemmatization, stopword removal, and frequency-based filtering.

## Installation

1. Install the required Python packages:
   ```
   conda env create -f environment.yml -n stream_etm
   conda activate stream_etm_base
   ```

   alternative : 

    `conda create -n stream_etm_base python==3.10`

    `conda activate stream_etm_base`

    `conda install -c conda-forge r-base rpy2`

    `pip install numpy seaborn pandas spacy`

    `pip install nltk matplotlib torch sentence-transformers plotly gensim jupyter`

2. Install the R package `ocp` for change point detection:
    Install R if you haven't already. You can download it from CRAN.
    Install the ocp package in R:
    ```
    R -e "install.packages('ocp')"
    ```
    or R -e "install.packages('ocp',repos = 'http://cran.us.r-project.org')" in command-line

    Ensure that R is accessible from your system's PATH environment variable so that rpy2 can interface with it.
    ```bash
    which R
    ```

3. Install SpaCy language models:
    ```
    python -m spacy download en_core_web_lg
    python -m spacy download en_core_web_md
    ```

4. Download NLTK stopwords:
    ```
    python -m nltk.downloader punkt punkt_tab stopwords 
    ```


## Usage

### Vocabulary Preloading

Before running the application, you need to preload the vocabulary and embeddings using the `vocab_loading.py` script:

```python
python -m pipelines.vocab_loading
```

### Configuration

Configure the application using a YAML file. An example configuration file is provided in `config/config_custom.yaml` 

### Preparing Your Data

- Create CSV files for each chunk of documents you want to process.
- Ensure the **news** column is present and contains the text for each document.
- Ensure the **keywords** and **headlines** columns are present, even if None values. 

Example CSV structure:

| id | date       | news                              | keywords                | headlines                     |
|----|------------|-----------------------------------|-------------------------|-------------------------------|
| 1  | 2022-01-01 | This is the first document.       | document                | This is first                 |
| 2  | 2022-01-02 | This is the second document.      | document                | This is second                |
| 3  | 2022-01-03 | Breaking news in the tech sector. | news, tech, sector      | Breaking news                 |
| 4  | 2022-01-04 | Market trends are shifting rapidly.| market, trends          | Market trends shifting        |
| ...| ...        | ...                               | ...                     | ...                           |

Example with 20 newsgroups text dataset is provided:

```python
python -m pipelines.data_generation
```

### Running the Application on a Corpus of Documents

To process all CSV files in the `data/documents` directory, you can use the provided `run_custom.sh` script. Make sure the script is executable:
```
chmod +x run.sh
./run.sh
```

### Preprocessing

The preprocessing module includes functions for lemmatization, stopword removal, and frequency-based filtering. Customize the preprocessing steps in `preprocessing.py`.

### Training and Merging

The `BasicTrainer` class in `trainer.py` handles the training of the ETM model. The `merge_etm_models` function in `merge.py` merges new topic models with existing ones.

### Change Point Detection

The `ocpd.py` module includes functions for running OCPD analysis on topic distributions. Configure the threshold for change point detection in the YAML configuration file.

## Results

After running the application, the results will be saved in the `runs` directory. Each run will have its own subdirectory named after the configuration file used. The following files will be generated:

- `doc_topic_distribution.csv`: Contains the document-topic assignments.
- `topic_over_time.csv`: Shows the topic proportions over time.
- `topics_representation.csv`: Lists the top words for each topic.
- `ocpd.csv`: Contains the detected change points for each topics over time.

## Visualization

You can visualize the results using the `analysis.ipynb` Jupyter notebook. This notebook provides various plots to analyze the topic proportions over time and the detected change points.


## Acknowledgements

- The ETM implementation is based on the paper "Topic Modeling in Embedding Spaces" by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei.
- The OCPD implementation uses the R package `ocp`.

For more information, please refer to the documentation and comments within the code and to the associated paper *Merging Embedded Topics with Optimal
Transport for Online Topic Modeling on Data Streams* (https://arxiv.org/pdf/2504.07711).
