from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import os

# Page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Header
st.title('📊 Sentiment Analysis')
st.subheader('Analyze text data for sentiment: Positive, Negative, or Neutral')

# Sentiment Analysis Function
def analyze_sentiment(polarity):
    if polarity > 0.1:  # Adjust threshold as needed
        return 'Positive'
    elif polarity < -0.1:  # Adjust threshold as needed
        return 'Negative'
    else:
        return 'Neutral'

# Text Analysis Section
with st.expander('📋 Analyze Text'):
    text = st.text_input('Enter your text:')
    if text:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment = analyze_sentiment(polarity)
        st.success(f'Sentiment: **{sentiment}**')

    pre = st.text_input('Clean your text:')
    if pre:
        cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                       stopwords=True, lowercase=True, numbers=True, punct=True)
        st.info(f'Cleaned Text: {cleaned_text}')

# CSV or Excel Analysis Section
with st.expander('📁 Analyze CSV/Excel'):
    upl = st.file_uploader('Upload your CSV or Excel file')
    
    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    if upl:
        file_extension = os.path.splitext(upl.name)[1]
        try:
            # Read the file based on its extension
            if file_extension == ".csv":
                df = pd.read_csv(upl)
            elif file_extension in [".xls", ".xlsx"]:
                df = pd.read_excel(upl)
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

            # Let user select the column containing text data
            column_name = st.selectbox('Select the column with text data:', df.columns)
            df['score'] = df[column_name].apply(score)
            df['analysis'] = df['score'].apply(analyze_sentiment)
            st.write(df.head(10))

            # Cache the conversion to CSV for download
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)

            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='sentiment_results.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: small;">
        &copy; Vinoth. All rights reserved.
    </div>
    """, 
    unsafe_allow_html=True
)
