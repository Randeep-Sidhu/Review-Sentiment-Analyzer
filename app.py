import streamlit as st
from transformers import pipeline
import pandas as pd

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

st.set_page_config(page_title="Review Sentiment Analyzer", layout="wide")
st.title("📝 Review Sentiment & Topic Analyzer")
st.markdown("""
Paste a product review or upload a CSV file with a **'review'** column.  
The app analyzes **sentiment** (positive, negative) and detects **topics** (Price, Quality, Delivery).  
""")

# --- Helper functions ---
def detect_topics(review_text):
    topics = []
    review_lower = review_text.lower()
    if any(word in review_lower for word in ["price", "cost", "expensive", "cheap"]):
        topics.append("Price")
    if any(word in review_lower for word in ["quality", "build", "material", "durable", "broken"]):
        topics.append("Quality")
    if any(word in review_lower for word in ["delivery", "shipping", "late", "arrived", "courier"]):
        topics.append("Delivery")
    return ", ".join(topics) if topics else "None detected"

def analyze_review(review):
    sentiment_result = sentiment_model(review)[0]
    sentiment = sentiment_result['label']
    score_raw = sentiment_result['score']
    score = min(round(score_raw, 2), 0.99)  # Cap score at 0.99
    topics = detect_topics(review)
    return sentiment, score, topics

def sentiment_color(sentiment):
    if sentiment.lower() == "positive":
        return "green"
    elif sentiment.lower() == "negative":
        return "red"
    else:
        return "orange"

def interpret_confidence(score):
    if score >= 0.85:
        return f"✅ High ({score})"
    elif score >= 0.60:
        return f"⚠️ Medium ({score})"
    else:
        return f"❌ Low ({score})"

# --- Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input")
    review_input = st.text_area("Paste your product review here", height=150)
    uploaded_file = st.file_uploader("Or upload a CSV file with a 'review' column", type=["csv"])
    analyze_clicked = st.button("Analyze")

with col2:
    st.subheader("Results")

    if analyze_clicked:
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                if 'review' not in df.columns:
                    st.error("CSV file must have a 'review' column.")
                else:
                    with st.spinner(f"Analyzing {len(df)} reviews..."):
                        sentiments = []
                        confidences = []
                        topics_list = []

                        for rev in df['review']:
                            s, c, t = analyze_review(str(rev))
                            sentiments.append(s)
                            confidences.append(interpret_confidence(c))
                            topics_list.append(t)

                        df['Sentiment'] = sentiments
                        df['Confidence'] = confidences
                        df['Topics'] = topics_list

                        # Only keep relevant columns for output
                        df_output = df[['review', 'Sentiment', 'Confidence', 'Topics']]

                    st.success("✅ Analysis complete!")

                    # Sentiment summary
                    st.markdown("### Sentiment Summary")
                    sentiment_counts = df['Sentiment'].value_counts()
                    for sentiment in ['POSITIVE', 'NEGATIVE']:
                        count = sentiment_counts.get(sentiment, 0)
                        color = sentiment_color(sentiment)
                        st.markdown(
                            f"- <span style='color:{color}; font-weight:bold'>{sentiment.title()}: {count}</span>",
                            unsafe_allow_html=True
                        )

                    # Show table
                    st.dataframe(df_output)

                    # Download button
                    csv = df_output.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download analysis results as CSV",
                        data=csv,
                        file_name='review_analysis_results.csv',
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"Error processing CSV file: {e}")

        elif review_input.strip() != "":
            with st.spinner("Analyzing review..."):
                sentiment, confidence_score, topics = analyze_review(review_input)

            color = sentiment_color(sentiment)
            confidence_label = interpret_confidence(confidence_score)

            st.markdown(f"**Sentiment:** <span style='color:{color}; font-weight:bold'>{sentiment}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {confidence_label}")
            st.markdown(f"**Topics Discussed:** {topics}")

            st.caption("ℹ️ Confidence shows how certain the AI is about the sentiment prediction.")

        else:
            st.info("Please paste a review or upload a CSV file to analyze.")
