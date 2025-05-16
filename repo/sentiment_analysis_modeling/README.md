# Sentiment Analysis Modeling for Movie Reviews

A functional approach to sentiment analysis of movie reviews using transformer models, Retrieval Augmented Generation (RAG), hybrid approaches and fine-tuning.

Note: Models and datasets are not included in the repository to reduce its size.  
However, they will be automatically downloaded when running the code.  
You can find the complete implementation at the following link:
https://drive.google.com/file/d/1OFoj1enMY0XWfWZZR0MSCWsGPvZhKi8U/view?usp=sharing

## Overview

This project implements and evaluates multiple approaches for sentiment classification on movie reviews:

- Transformer-based classification using DistilBERT
- Retrieval Augmented Generation (RAG) using vector similarity
- Hybrid approach combining transformer confidence scores with RAG
- Fine-tuning of the transformer model

### Performance Comparison

| Approach                                         | Accuracy |
|--------------------------------------------------|----------|
| Pretrained transformer (DistilBERT)              | 88%      |
| RAG-style retrieval from similar reviews         | 64%      |
| Hybrid model (DistilBERT + retrieval)            | 86%      |
| Finetuned DistilBERT (with limited input length) | 85%      |

The hybrid approach uses the transformer model for high-confidence predictions (76.8% of cases) and falls back to RAG for ambiguous reviews (23.2%), providing a good balance between accuracy and robustness.

## Project Structure

```
├── data/                                # Data storage
│   ├── raw/                             # Raw IMDB dataset
│   └── vectorial_database/              # ChromaDB vector database
├── model/                               # Saved models
│   ├── classification_model/            # DistilBERT sentiment model
│   └── embedding_model/                 # Sentence transformer embedding model
├── results/                             # Performance metrics
│   ├── only_transformer_model/          # Results for transformer-only approach
│   ├── only_rag_one_retrieval/          # Results for RAG with single retrieval
│   ├── only_rag_three_retrievals/       # Results for RAG with three retrievals
│   └── threshold_hybrid_model_with_rag/ # Results for hybrid approach
├── src_eda/                             # Exploratory data analysis
├── src_preparation/                     # Data and model preparation scripts
├── src_testing_and_performance/         # Evaluation scripts
└── Fine_tuning_DistilBERT_on_IMDB_using_TensorFlow_Transformers.ipynb     # Fine-tuning notebook
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Download the IMDB dataset and prepare it for analysis:

```bash
python src_preparation/download_data.py
```

### 2. Exploratory Data Analysis

```bash
python src_eda/eda.py
```

### 3. Models Download

Download the pre-trained DistilBERT model:

```bash
python src_preparation/download_classification_model.py
```

Download the pre-trained SentenceTransformer model:

```bash
python src_preparation/download_embedding_model.py
```

### 4. Vector Database

Create the vector database:

```bash
python src_preparation/create_vector_db.py
```


### 5. Model testing

```bash
# Test transformer model
python src_testing_and_performance/test_model_classification_model.py

# Test transformer model with RAG
python src_testing_and_performance/test_model_classification_model_with_rag.py
```

### 6. Evaluation

Evaluate different approaches:

```bash
# Evaluate transformer model only
python src_testing_and_performance/evaluation_transformer_model.py

# Evaluate RAG approach with three retrievals
python src_testing_and_performance/evaluation_rag_three_retrievals.py

# Evaluate hybrid approach
python src_testing_and_performance/evaluation_hybrid_model_with_rag.py
```

## Model

This solution uses `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face's Transformers library, fine-tuned on the Stanford Sentiment Treebank dataset, and enhanced with a retrieval mechanism for improved robustness.
