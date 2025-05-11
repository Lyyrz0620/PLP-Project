# Steam Reviews Analysis System

This project presents a modular NLP system for analyzing and interacting with user reviews from the [Steam Reviews Dataset](https://www.kaggle.com/datasets/andrewmvd/steam-reviews). The system includes sentiment classification, topic modeling, and a question-answering (QA) module, all accessible through an interactive UI.

---

## 🔧 Project Structure

```

PLP PROJECT/
├── Data/
│   └── Data_cleaning.ipynb               # Preprocessing of raw Steam reviews
├── QA module/
│   ├── qa_model.py                       # QA logic module for use in front-end (UI) interaction
│   └── QA.ipynb                          # QA module pipeline and demonstration
├── SA Module/
│   └── sentiment_model_finetuned.ipynb   # Fine-tuning of sentiment classification model
├── TM Module/
│   └── Auto_Topic_Modeling.ipynb         # BERTopic-based topic modeling implementation
├── UI/
│   └── Web_final.ipynb                   # Gradio-based user interface
├── README.md                             # Project documentation

````

---

## 📌 Key Features

- **Sentiment Analysis**: Fine-tuned `twitter-roberta-base-sentiment` model for classifying reviews into Positive or Negative.
- **Topic Modeling**: Clusters user reviews into coherent topics using BERTopic.
- **Question Answering**: Supports semantic queries using a dense retrieval + reranking pipeline with FAISS and cross-encoders.
- **Interactive UI**: Gradio-powered interface for querying and exploring user feedback in real time.

---


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Lyyrz0620/PLP-Project.git
cd steam-reviews-nlp
````

### 2. Install Dependencies

This project is designed to run in **Google Colab**. All required packages can be installed at the beginning of each notebook using the following commands:

```python
# Core libraries
!pip install transformers datasets
!pip install sentence-transformers
!pip install faiss-cpu
!pip install bertopic
````

### 3. Prepare Dataset

Download the [Steam Reviews dataset](https://www.kaggle.com/datasets/andrewmvd/steam-reviews), and run `Data/Data_cleaning.ipynb` to preprocess and clean the reviews.

### 4. Run Individual Modules

Each module is standalone:

* **SA**: `sentiment model finetuned.ipynb`
* **TM**: `Auto Topic Modeling.ipynb`
* **QA**: `QA.ipynb`

### 5. Run the Front-End Interface

To launch the user interface, simply open and run all cells in the notebook: `UI/Web_final.ipynb`

This notebook contains the full interactive front-end implementation. It loads the pre-trained models and allows users to perform sentiment analysis, topic modeling, and QA in one place.

