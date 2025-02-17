import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "D:\\Software testing with Python batch 24\\Amazon_Sentiamte_Data4\\processed_reviews.csv "  # Update this with the correct path
df = pd.read_csv(file_path, encoding="latin1")

# Display basic info
print("Basic Info:\n", df.info())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Convert 'Score' column to numeric (if not already)
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

# Drop missing values in 'Score'
df = df.dropna(subset=['Score'])

# Summary statistics
print("\nScore Statistics:\n", df['Score'].describe())

# Set style
sns.set_style("whitegrid")

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['Score'], bins=10, kde=True, color="skyblue")
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Box plot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Score'], color="lightcoral")
plt.title("Score Boxplot")
plt.show()

# Violin Plot (combination of boxplot & KDE)
plt.figure(figsize=(6, 4))
sns.violinplot(x=df['Score'], color="lightgreen")
plt.title("Score Violin Plot")
plt.show()

# Pie Chart for Score Distribution
score_counts = df['Score'].value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(score_counts, labels=score_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title("Score Percentage Distribution")
plt.show()

# Bar Plot for Score Frequency
plt.figure(figsize=(8, 5))
sns.barplot(x=score_counts.index, y=score_counts.values, palette="muted")
plt.title("Score Frequency")
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()
