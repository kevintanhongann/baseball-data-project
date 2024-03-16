import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Load the hitting and pitching data
hitting_data = pd.read_csv('baseball_hitting.csv')
pitching_data = pd.read_csv('baseball_pitcher.csv')

# Drop rows with any empty cells
hitting_data.dropna(inplace=True)
pitching_data.dropna(inplace=True)

# Calculate OPS for each player
hitting_data['OPS'] = hitting_data['On-base Percentage'] + hitting_data['Slugging Percentage']

# Predicting Runs using other hitting stats
X = hitting_data[['At-bat', 'Hits', 'home run', 'run batted in']]
y = hitting_data['Runs']
model = LinearRegression().fit(X, y)
predictions = model.predict(X)

# Histogram of home runs
fig = px.histogram(hitting_data, x='home run', title='Distribution of Home Runs')
fig.show()

# Scatter Plot of At-bat vs Runs
fig = px.scatter(hitting_data, x='At-bat', y='Runs', title='Scatter Plot of At-bat vs Runs')
fig.show()

# Line Chart of Games vs Hits
fig = px.line(hitting_data, x='Games', y='Hits', title='Line Chart of Games vs Hits')
fig.show()

# Bar Chart of Home Runs by Player
fig = px.bar(hitting_data, x='Player name', y='home run', title='Bar Chart of Home Runs by Player')
fig.show()

# Box Plot of Strikeouts
fig = px.box(pitching_data, y='Strikeouts', title='Box Plot of Strikeouts')
fig.show()

# Heatmap of Hitting Data Correlation
name_col = 'Player name'
position_col = 'position'
hitting_data = hitting_data.drop(columns=[name_col,position_col])
corr = hitting_data.corr()
fig = px.imshow(corr, text_auto=True, title='Heatmap of Hitting Data Correlation')
fig.show()

# Actual vs Predicted Runs
fig = px.scatter(x=range(len(y)), y=y, labels={'x':'Index', 'y':'Runs'}, title='Actual Runs')
fig.add_scatter(x=range(len(predictions)), y=predictions, name='Predicted')
fig.show()
