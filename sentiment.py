import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

df = pd.read_csv("twitter_training.csv", header=None)
df.columns = ["id", "brand", "sentiment", "text"]
output_lines =[]

# Sentiment counts
sentiment_counts = df["sentiment"].value_counts()
output_lines.append("Overall Sentiment Counts:\n")
output_lines.append(sentiment_counts.to_string())
output_lines.append("\n\n")

print("Overall Sentiment Counts:\n")
print(sentiment_counts)

#Plot sentiment distribution
sns.countplot(x="sentiment", data=df)
plt.title("Overall Sentiment Distribution")
plt.savefig("sentiment_distribution.png")
plt.show()

#Brand vs Sentiment table
brand_sentiment = pd.crosstab(df["brand"], df["sentiment"])
output_lines.append("Brand vs Sentiment:\n")
output_lines.append(brand_sentiment.to_string())
output_lines.append("\n\n")

print("\nBrand vs Sentiment:\n")
print(brand_sentiment)

brand_sentiment.plot(kind="bar", stacked=True)
plt.title("Sentiment by Brand")
plt.xlabel("Brand")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("brand_sentiment.png")
plt.show()

#Clean text
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z ]", "", text)
    return text.lower()

df["clean_text"] = df["text"].apply(clean_text)

#Common words
all_words = " ".join(df["clean_text"])
words = all_words.split()

common_words = Counter(words).most_common(10)
output_lines.append("Top 10 Common Words:\n")
for word, count in common_words:
    output_lines.append(f"{word}: {count}")

print("\nTop 10 Common Words:")
print(common_words)

# Save all printed output into a text file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("\nOutput saved to output.txt")
print("Charts saved as sentiment_distribution.png and brand_sentiment.png")