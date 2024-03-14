import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


# Load the hitting and pitching data
hitting_data = pd.read_csv('baseball_hitting.csv')
pitching_data = pd.read_csv('baseball_pitcher.csv')


# Check for missing values
print(hitting_data.isnull().sum())
print(pitching_data.isnull().sum())

# Fill missing values or drop rows/columns if necessary
# hitting_data.fillna(0, inplace=True)
# pitching_data.fillna(0, inplace=True)

# Drop rows with any empty cells
hitting_data.dropna(inplace=True)
pitching_data.dropna(inplace=True)

# Summary statistics for hitting data
print(hitting_data.describe())

# Summary statistics for pitching data
print(pitching_data.describe())

# Histogram of home runs
hitting_data['home run'].hist()
plt.title('Distribution of Home Runs')
plt.xlabel('Home Runs')
plt.ylabel('Frequency')
plt.show()


# Correlation matrix for hitting data
# name_col = 'Player name'
# position_col = 'position'
# hitting_data = hitting_data.drop(columns=[name_col,position_col])
# print(hitting_data.corr())


# Calculate OPS for each player
hitting_data['OPS'] = hitting_data['On-base Percentage'] + hitting_data['Slugging Percentage']



# Predicting Runs using other hitting stats
X = hitting_data[['At-bat', 'Hits', 'home run', 'run batted in']]
y = hitting_data['Runs']
model = LinearRegression().fit(X, y)


# Scatter Plots: Useful for showing the relationship between two variables.
plt.scatter(hitting_data['At-bat'], hitting_data['Runs'])
plt.xlabel('At-bat')
plt.ylabel('Runs')
plt.title('Scatter Plot of At-bat vs Runs')
plt.show()

# Line Charts: Ideal for visualizing trends over time or ordered categories.
plt.plot(hitting_data['Games'], hitting_data['Hits'])
plt.xlabel('Games')
plt.ylabel('Hits')
plt.title('Line Chart of Games vs Hits')
plt.show()

# Bar Charts: Good for comparing different groups or categories.
plt.bar(hitting_data['Player name'], hitting_data['home run'])
plt.xlabel('Player Name')
plt.ylabel('Home Runs')
plt.title('Bar Chart of Home Runs by Player')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()

# Histograms: Show the distribution of a numerical variable.
plt.hist(pitching_data['Earned run Average'])
plt.xlabel('Earned Run Average (ERA)')
plt.ylabel('Frequency')
plt.title('Histogram of ERA')
plt.show()

# Box Plots: Summarize the distribution of a dataset and identify outliers.
plt.boxplot(pitching_data['Strikeouts'])
plt.ylabel('Strikeouts')
plt.title('Box Plot of Strikeouts')
plt.show()

# Heatmaps: Visualize patterns of correlation or co-occurrence.
name_col = 'Player name'
position_col = 'position'
hitting_data = hitting_data.drop(columns=[name_col,position_col])
corr = hitting_data.corr()
sns.heatmap(corr, annot=True)
plt.title('Heatmap of Hitting Data Correlation')
plt.show()


# Predicting Runs using other hitting stats
X = hitting_data[['At-bat', 'Hits', 'home run', 'run batted in']]
y = hitting_data['Runs']
model = LinearRegression().fit(X, y)

# Get predicted values
predictions = model.predict(X)

plt.scatter(range(len(y)), y, label='Actual')
plt.scatter(range(len(predictions)), predictions,
            label='Predicted', color='red')
plt.legend()
plt.title('Actual vs Predicted Runs')
plt.show()
