# Review Sentiment Analyzer

**Streamlit app for analyzing customer product reviews using state-of-the-art NLP models**

---

## 🚀 Project Overview

This interactive web app allows users to analyze product reviews for:

- **Sentiment detection**: Classifies reviews as positive or negative with confidence scores.
- **Topic identification**: Extracts key discussion points such as Price, Quality, and Delivery.
- **Batch processing**: Analyze single reviews or upload CSV files for bulk analysis.
- **Downloadable results**: Export the analyzed data with sentiments, confidence levels, and topics for further insights.

This project demonstrates practical experience with:

- Transformer-based sentiment analysis pipelines (Hugging Face Transformers)
- Building clean, user-friendly UIs with Streamlit
- Data handling with Pandas
- Packaging projects with `requirements.txt` for reproducibility

---

## 🎯 Why This Project?

- Shows ability to build AI-powered applications end-to-end
- Practical use of NLP techniques for real-world business problems (e-commerce/product feedback)
- Experience with Python web frameworks and cloud-ready deployment

---

## 💡 Features

- Paste or upload product reviews (CSV file with 'review' column)
- Clear visualization of sentiment with emoji and confidence indicators
- Topic detection for focused feedback analysis
- Download analyzed data as CSV for reporting or further processing

---

## ⚙️ Requirements

- Python 3.8+
- See detailed dependencies in [`requirements.txt`](requirements.txt)

---

## 🏃‍♂️ Running the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
