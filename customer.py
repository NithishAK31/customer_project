import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("./sentiment_analysis_feedback_3000_updated.csv")
# df = pd.read_csv("./negative_feedback.csv")

# Function to calculate sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis to feedback texts
df['Sentiment'] = df['Customer Feedback Text'].apply(get_sentiment)

# Count occurrences of each sentiment type
sentiment_counts = df['Sentiment'].value_counts()

# Plot the sentiment counts in a bar chart
plt.figure(figsize=(8,6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Analysis of Customer Feedback')
plt.xlabel('Sentiment')
plt.ylabel('Number of Feedback')
plt.xticks(rotation=0)
plt.show()

# If you want to save the separated data into CSVs for each sentiment
# df[df['Sentiment'] == 'Positive'].to_csv('positive_feedback.csv', index=False)
# df[df['Sentiment'] == 'Negative'].to_csv('negative_feedback.csv', index=False)
# df[df['Sentiment'] == 'Neutral'].to_csv('neutral_feedback.csv', index=False)

df.to_csv('sentiment_feedback.csv', index=False)