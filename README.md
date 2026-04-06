# Community Stress Tracker AI

Community Stress Tracker AI is a data-driven natural language processing project that analyzes textual inputs to detect emotional states and estimate stress levels. The system aggregates these signals across multiple inputs to provide a broader view of stress patterns, with the goal of supporting meaningful insights for social good.

---

## Overview

This project was developed as part of a beginner-friendly hackathon, **Predict4Good**, which focused on building predictive solutions that address real-world problems. It explores how machine learning and structured logic can be used to interpret unstructured text and transform it into actionable insights.

The system is designed to:
- Identify emotional signals from text using a trained model (DistilBERT)
- Convert these signals into quantitative stress scores
- Normalize and aggregate results to reflect overall stress trends in a community

By combining prediction with simple aggregation logic, the project demonstrates how data-driven approaches can contribute to understanding mental well-being in everyday contexts.

---

## Problem Statement

Mental health signals are often expressed through informal and messy text, such as social media posts or personal messages. However, these signals are difficult to analyze at scale and are rarely translated into interpretable metrics.

This project addresses this gap by providing a method to:
- Detect emotional patterns from text  
- Quantify stress levels in a consistent way  
- Aggregate signals across multiple inputs to estimate community-level stress trends

By extending analysis beyond individual posts, the system enables a broader understanding of how stress manifests across groups or shared spaces. This can help identify periods of heightened distress, recurring emotional patterns, or shifts in collective well-being over time.

Such insights can be valuable to researchers, organizations, and platforms that aim to monitor and respond to mental health signals at scale. In this way, the project demonstrates how predictive and data-driven approaches can contribute to early awareness and support efforts that promote community well-being.

---

## Approach and Prediction Logic

The system follows a structured, data-driven pipeline:

1. Text preprocessing to standardize input
2. Conversion of the textual labels (*Suicidal, Depressed, Anxious, Frustrated, Others*) into numeric form using label encoding
3. Tokenization using a pretrained DistilBERT tokenizer  
4. Emotion classification using a fine-tuned DistilBERT model  
5. Mapping predicted labels to stress scores (1–5 scale)  
6. Normalizing scores to a 0–10 range  
7. Aggregating scores to estimate overall stress levels  

This approach combines machine learning with simple logic-based aggregation, making it both interpretable and practical for real-world use.

---

## Model and Implementation

The project uses DistilBERT for sequence classification. The model is trained on a labeled dataset named `MentalDistress`[https://data.mendeley.com/datasets/b42wr437hg/2], a curated and annotated set of English dataset categorized into five psychological states and containing mental health-related categories, and optimized using class-weighted loss to address imbalance.

The implementation focuses on:
- Clear and modular code structure  
- Reproducible preprocessing and evaluation steps  
- Efficient inference using saved model weights  
- Deployment through a Streamlit interface for real-time interaction  

The system is fully functional and allows users to input text and receive immediate predictions along with stress analysis.

---

## Features

- Transformer-based emotion classification
- Stress scoring and normalization
- Aggregation of multiple inputs into a single stress indicator
- Interactive web interface using Streamlit

---

## Model Deployment

Rather than focusing solely on classification, this project introduces an additional layer of interpretation by transforming categorical outputs into a continuous stress scale and aggregating them across inputs.

This allows the system to move beyond individual predictions and provide a broader perspective on community-based stress trends, making the solution more meaningful and aligned with real-world applications.

---

## Project Structure
```
├── app.py # Streamlit application
├── distilbert_emotion_model.pth # Trained model weights
├── label_mapping.json # Label encoding mapping
├── mental_distress_test_set.csv # Test dataset
├── Mental_Distress_Dataset-original.csv # Original raw dataset
├── Mental_Distress_Dataset_updated.csv # Updated and preprocessed dataset with null rows dropped
├── community_stress_fullTraining_notebook.ipynb # Training and experimentation from scratch
├── community_stress_modelLoading_notebook.ipynb # Training and experimentation using saved trained model
├── README.md
```


---

## Installation

Install the required dependencies in your local terminal:
```
pip install streamlit transformers torch emoji emoticon_fix pandas scikit-learn
```

---

## Running the Application

To launch the application: `streamlit run app.py`

---

## Stress Scoring Scheme

| Emotion     | Score |
|------------|------|
| Suicidal   | 5    |
| Depressed  | 4    |
| Anxious    | 3    |
| Frustrated | 2    |
| Others     | 1    |

Scores are normalized to a 0–10 scale for easier interpretation.

---

## Use Cases

- Monitoring stress patterns from textual inputs  
- Supporting exploratory analysis in mental health research  
- Building tools that promote awareness of emotional well-being  
- Demonstrating predictive and data-driven solutions for social impact  

---

## Limitations

The system relies solely on textual input and may not capture the full context of an individual’s mental state. Predictions should be interpreted as indicative rather than definitive.

---

## Disclaimer

This project is created for the hackathon purpose, intended for educational and research purposes only. It is not a substitute for professional mental health advice, diagnosis, or treatment.

---

## Future Work

- Incorporating time-based tracking for trend analysis
- Extending to community-level dashboards
- Integrating additional datasets for improved generalization
- Exploring explainable AI techniques for transparency 

---

**Project Contributors:** Adrika Chowdhury, Tustee Mazumdar
