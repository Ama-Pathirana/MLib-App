import streamlit as st
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import matplotlib.pyplot as plt

# Initialize Spark
spark = SparkSession.builder.appName("Lyrics Genre Classifier").getOrCreate()

# Load trained model
model = PipelineModel.load("final_model")

# Title
st.title("🎵 Music Genre Classifier")
st.write("Paste your song lyrics below and click 'Predict Genre' to see the results.")

# Input text
lyrics_input = st.text_area("Paste your lyrics here", height=300)

if st.button("Predict Genre"):
    if lyrics_input.strip() == "":
        st.warning("Please enter some lyrics.")
    else:
        # Create a DataFrame from input
        df = pd.DataFrame([[lyrics_input]], columns=["lyrics"])
        sdf = spark.createDataFrame(df)

        # Predict
        predictions = model.transform(sdf)
        result = predictions.select("prediction").collect()[0][0]

        # You should map numerical predictions to actual genre labels if needed
        genre_labels = ['pop', 'country', 'blues', 'jazz', 'reggae', 'rock', 'hip hop', 'your_genre']
        predicted_genre = genre_labels[int(result)]

        # Display prediction
        st.success(f"Predicted Genre: {predicted_genre}")

        # Bar chart for visualization
        genre_distribution = [0] * len(genre_labels)
        genre_distribution[int(result)] = 1

        st.bar_chart(pd.Series(genre_distribution, index=genre_labels))
