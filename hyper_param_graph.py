import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('tf_idf.csv')  # Replace with your actual file name

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check the column names and the first few rows of the DataFrame
print(df.columns)
print(df.head())

# Create a color map based on accuracy
norm = plt.Normalize(df['Accuracy'].min(), df['Accuracy'].max())
colors = plt.cm.viridis(norm(df['Accuracy']))

# Convert 'K' to integers if necessary
df['K'] = df['K'].astype(int)

# Plot K vs Order
plt.figure(figsize=(12, 6))
scatter3 = plt.scatter(df['K'], df['Order'], c=colors, s=100, alpha=0.7, edgecolors='w')
plt.title('K vs Order')
plt.xlabel('K')
plt.ylabel('Order')

# Set x-axis ticks to be integers
plt.xticks(df['K'].unique())  # Set x-ticks to unique values of K

# Add colorbar
cbar3 = plt.colorbar(scatter3, label='Accuracy')  # Associate colorbar with scatter plot
plt.grid()
plt.show()
