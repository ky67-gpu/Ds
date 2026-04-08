import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
url = 'https://en.wikipedia.org/wiki/List_of_Asian_countries_by_area'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.
(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/573.36'}
response = requests.get(url, headers=headers)
page = response.text
print(page)
soup = BeautifulSoup(page, 'html.parser')
table = soup.find('table')
print(table)
SrNo = []
Country = []
Area = []
rows = table.find('tbody').find_all('tr')
for row in rows:
cells = row.find_all('td')
if cells:
SrNo.append(cells[0].get_text().strip('\n'))
Country.append(cells[1].get_text().strip('\xa0').strip('\n'))
Area.append(cells[3].get_text().strip('\n').replace(',', ''))
df = pd.DataFrame()
df['SrNo'] = SrNo
df['Country'] = Country
df['Area'] = Area
df.head(10)
OUTPUT: DataFrame df.head(10) shows: Id, Username, Email columns with 10 rows of Asian countries data.

Part 2: JSON Scraping (JSONPlaceholder)
CODE:

import json
from urllib.request import urlopen
import pandas as pd
url = 'https://jsonplaceholder.typicode.com/users'
page = urlopen(url)
data = json.loads(page.read())
Id = []
Username = []
Email = []
for item in data:
if 'id' in item.keys():
Id.append(item['id'])
else:
Id.append('NA')
if 'username' in item.keys():
Username.append(item['username'])
else:
Username.append('NA')
if 'email' in item.keys():
Email.append(item['email'])
else:
Email.append('NA')
df = pd.DataFrame()
df['Id'] = Id
df['Username'] = Username
df['Email'] = Email
df.head(10)
df.info()
OUTPUT: df.head(10): 10 rows with Id, Username, Email. df.info(): RangeIndex 10 entries, 3 non-null

columns.

Practical 2 – EDA on Titanic Dataset
CODE:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
titanic_df = pd.read_csv('Titanic Dataset.csv')
titanic_df.head()
titanic_df.info()
titanic_df.isnull().sum()
titanic_df.describe()
titanic_cleaned = titanic_df.drop(['PassengerId','Name','Ticket','Fare','Cabin'], axis=1)
titanic_cleaned.head()
titanic_cleaned['Age'] = titanic_cleaned['Age'].fillna(
titanic_cleaned.groupby('Pclass')['Age'].transform('mean')
)
titanic_cleaned.isnull().sum()
# Plot 1: Survival by Gender
sns.catplot(x='Sex', hue='Survived', kind='count', data=titanic_cleaned)
# Count by Sex and Survived
titanic_cleaned.groupby(['Sex','Survived'])['Survived'].count()
# Heatmap: Gender vs Survival
group1 = titanic_cleaned.groupby(['Sex','Survived'])
gender_survived = group1.size().unstack()
sns.heatmap(gender_survived, annot=True, fmt='d')
# Heatmap: Pclass vs Survival
group2 = titanic_cleaned.groupby(['Pclass','Survived'])
pclass_survived = group2.size().unstack()
sns.heatmap(pclass_survived, annot=True, fmt='d')
# Violin Plot: Age distribution by Sex and Survival
sns.violinplot(x='Sex', y='Age', hue='Survived', data=titanic_cleaned, split=True)
# Pearson Correlation Heatmap
titanic_corr = titanic_cleaned.drop(['Sex','Embarked'], axis=1)
titanic_corr.corr(method='pearson')
sns.heatmap(titanic_corr.corr(method='pearson'), annot=True, vmax=1)
OUTPUT: titanic_df.head(): 5 rows with 12 columns. isnull().sum(): Age=177, Cabin=687, Embarked=2.

Visualizations: Count plot (sex vs survival), heatmaps, violin plot, Pearson correlation heatmap.

Practical 3 – EDA on MTCars Dataset
CODE:

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
os.getcwd()
df = pd.read_csv('C:\\Users\\User\\Downloads\\MTCARS.csv')
print(df.head())
print(df.tail())
df.info()
print(df.isnull().sum())
print(df.describe())
print(df.shape)
# Clean data
df_cleaned = df.drop(['model'], axis=1)
df_cleaned['mpg'] = df_cleaned['mpg'].fillna(
df_cleaned.groupby('cyl')['mpg'].transform('mean')
)
# Plot 1: Distribution of Cylinders
plt.figure(figsize=(8, 5))
sns.countplot(x='cyl', data=df_cleaned)
plt.title('Distribution of Cylinders')
plt.show()
# Plot 2: MPG Distribution Histogram
plt.figure(figsize=(8, 5))
plt.hist(df_cleaned['mpg'], bins=10, edgecolor='black')
plt.title('MPG Distribution')
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.show()
# Plot 3: Horsepower Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df_cleaned['hp'])
plt.title('Horsepower Boxplot')
plt.show()
# Plot 4: Cylinders vs Transmission
sns.catplot(x='cyl', hue='am', kind='count', data=df_cleaned)
plt.title('Cylinders vs Transmission')
plt.show()
# Plot 5: MPG Distribution by Cylinders & Transmission (Violin)
plt.figure(figsize=(10, 6))
sns.violinplot(x='cyl', y='mpg', hue='am', data=df_cleaned, split=True)
plt.title('MPG Distribution by Cylinders & Transmission')
plt.show()
# Plot 6: Weight vs MPG Scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wt', y='mpg', hue='cyl', size='hp', data=df_cleaned)
plt.title('Weight vs MPG (Colored by Cylinders, Size by HP)')
plt.show()
# Plot 7: Correlation Heatmap
mtcars_corr = df_cleaned.corr(method='pearson')
print('\nCorrelation Matrix:')
print(mtcars_corr)
plt.figure(figsize=(12, 8))
sns.heatmap(mtcars_corr, annot=True, vmax=1, vmin=-1, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
OUTPUT: df.shape: (32, 12). Plots generated: Cylinder distribution bar chart, MPG histogram, Horsepower

boxplot, Cylinders vs Transmission catplot, Violin plot (MPG by cyl & am), Scatter plot (Weight vs MPG),

Correlation heatmap.

Practical 4 – Linear Regression
Part 1: Experience vs Salary Prediction
CODE:

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Generate sample regression data
x, y, coef = datasets.make_regression(
n_samples=100,
n_features=1,
n_informative=1,
noise=10,
coef=True,
random_state=
)
# Scale to realistic ranges
x = np.interp(x, (x.min(), x.max()), (0, 20))
y = np.interp(y, (y.min(), y.max()), (20000, 150000))
# Plot training data
plt.plot(x, y, '.', label='Training Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.legend()
plt.show()
# Train linear regression model
reg_model = LinearRegression()
reg_model.fit(x, y)
y_predicted = reg_model.predict(x)
# Plot with regression line
plt.plot(x, y, '.', label='Training Data')
plt.plot(x, y_predicted, '.', color='red', label='Predicted Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.legend()
plt.show()
OUTPUT: Scatter plot: Blue dots = training data, Red dots = predicted salary. Salary increases linearly from

~20,000 to ~1,50,000 over 0–20 years experience.

Part 2: Simulating Y = 10 + 7x + e
CODE:

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
# Generate random input values (100 samples)
x = np.random.rand(100, 1)
error = np.random.rand(100, 1)
# Linear model: Y = 10 + 7x + error
b0 = 10
b1 = 7
y = b0 + b1 * x + error
# Train model
reg_model = LinearRegression()
reg_model.fit(x, y)
y_predicted = reg_model.predict(x)
# Plot results
plt.plot(x, y_predicted, '.', color='black', label='Predicted Data')
plt.plot(x, y, '.', label='Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y')
plt.legend()
plt.show()
OUTPUT: Scatter plot: Blue dots = training data (with noise), Black dots = predicted line. Y ranges ~10–18 for

X in [0,1].

Practical 5 – Multiple Linear Regression (Boston Dataset)
CODE:

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
os.getcwd()
# Load dataset
boston_df = pd.read_csv('Boston.csv')
boston_df.head()
boston_df.info()
# Remove unnecessary index column
boston_df = boston_df.drop('Unnamed: 0', axis=1)
boston_df.info()
# Separate features and target
boston_x = pd.DataFrame(boston_df.iloc[:, :13])
boston_y = pd.DataFrame(boston_df.iloc[:, -1])
# Split into train/test (70/30)
X_train, X_test, Y_train, Y_test = train_test_split(
boston_x, boston_y, test_size=0.
)
print(f'X Train Size: {X_train.shape}')
print(f'X Test Size: {X_test.shape}')
print(f'Y Train Size: {Y_train.shape}')
print(f'Y Test Size: {Y_test.shape}')
# Create and train model
reg_model = LinearRegression()
reg_model.fit(X_train, Y_train)
# Predict
y_predicted = reg_model.predict(X_test)
Y_pred = pd.DataFrame(y_predicted, columns=['Predicted_Y'])
Y_pred.head()
# Plot actual vs predicted
plt.scatter(Y_test, Y_pred, c='green')
plt.xlabel('Actual Price (MEDV)')
plt.ylabel('Predicted Price')
plt.show()
OUTPUT: Train/Test split: (354,13) / (152,13). Y_pred.head(): [26.99, 40.27, 17.14, 23.66, 14.51]. Scatter plot

shows Actual vs Predicted MEDV values clustered near diagonal.

Practical 6 – KNN Classification (Breast Cancer)
CODE:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()
# Load dataset
breast_cancer_data = load_breast_cancer()
# Create feature DataFrame
X_df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
X_df.head()
X_df.info()
# Select 2 features
X_df = X_df[['mean area', 'mean compactness']]
X_df.head()
X_df.info()
# Create and encode target variable
Y_df = pd.Categorical.from_codes(breast_cancer_data.target,
breast_cancer_data.target_names)
print(Y_df)
Y_df = pd.get_dummies(Y_df, drop_first=True)
Y_df.info()
print(Y_df)
# Split dataset (75% train, 25% test)
X_train, X_test, Y_train, Y_test = train_test_split(
X_df, Y_df, random_state=1, test_size=0.25, shuffle=True
)
# Verify split sizes
X_test.info()
Y_test.info()
X_train.info()
Y_train.info()
# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, Y_train)
# Visualize actual test data
combined_df = pd.concat([X_test, Y_test], axis=1)
sns.scatterplot(x='mean area', y='mean compactness', hue='benign', data=combined_df)
# Predict and visualize
Y_pred = knn.predict(X_test)
plt.scatter(X_test['mean area'], X_test['mean compactness'],
c=Y_pred, cmap='coolwarm', alpha=0.7)
# Confusion Matrix
cf = confusion_matrix(Y_test, Y_pred)
print(cf)
tp, fn, fp, tn = confusion_matrix(Y_test, Y_pred, labels=[1,0]).reshape(-1)
print(tp, fn, fp, tn)
# Heatmap of Confusion Matrix
ax = plt.subplot()
sns.heatmap(cf, annot=True, ax=ax)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Malignant', 'Benign'])
ax.yaxis.set_ticklabels(['Malignant', 'Benign'])
OUTPUT: Confusion Matrix: [[42, 13], [9, 79]]. TP=79, FN=9, FP=13, TN=42. Heatmap shows

Malignant/Benign predicted vs actual.

Practical 7 – Decision Tree (Titanic)
CODE:

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
# Load dataset
titanic_df = pd.read_csv('train.csv')
titanic_df.info()
# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
x = titanic_df[features]
x.info()
# Target variable
Y = titanic_df['Survived']
Y.info()
# Encode categorical data
x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})
x.head()
# Fill missing Age values
x['Age'].fillna(x['Age'].median(), inplace=True)
# Check missing values
x.isnull().sum()
# Split dataset (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(
x, Y, test_size=0.2, random_state=
)
# Create Decision Tree model
dtmodel = DecisionTreeClassifier(
criterion='entropy',
max_depth=4,
random_state=
)
# Train and predict
dtmodel.fit(X_train, Y_train)
Y_pred = dtmodel.predict(X_test)
# Evaluate
print('Accuracy: ', accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
# Visualize Decision Tree
plt.figure(figsize=(18, 10))
plot_tree(
dtmodel,
feature_names=features,
class_names=['Not Survived', 'Survived'],
filled=True
)
plt.show()
OUTPUT: Accuracy: 0.8379. Classification report: precision 0.87/0.77, recall 0.88/0.76, f1-score 0.88/0.76.

Decision tree diagram with max_depth=4, splits on Sex, Pclass, Age, SibSp, Parch.

Practical 8 – MongoDB Basics
Q1: Staff Collection
INSERT:

use Institution
db.Staff.insertMany([
{empid:1, empname:'Rahul', salary:40000, designation:'Clerk'},
{empid:2, empname:'Anita', salary:55000, designation:'Manager'},
{empid:3, empname:'Suresh', salary:30000, designation:'Assistant'},
{empid:4, empname:'Meena', salary:45000, designation:'Accountant'},
{empid:5, empname:'Ravi', salary:70000, designation:'Manager'},
{empid:6, empname:'Kiran', salary:25000, designation:'Clerk'},
{empid:7, empname:'Pooja', salary:48000, designation:'Accountant'},
{empid:8, empname:'Arjun', salary:90000, designation:'Manager'},
{empid:9, empname:'Neha', salary:120000, designation:'Director'},
{empid:10, empname:'Amit', salary:35000, designation:'Assistant'}
])
QUERIES:

// Display all documents
db.Staff.find()
// Display only empid and designation
db.Staff.find({}, {empid:1, designation:1, _id:0})
// Sort in descending order of Salary
db.Staff.find().sort({salary:-1})
// Display Manager OR salary > 50000
db.Staff.find({
$or: [
{designation: 'Manager'},
{salary: {$gt: 50000}}
]
})
// Update Accountant salary to 45000
db.Staff.updateMany(
{designation: 'Accountant'},
{$set: {salary: 45000}}
)
// Remove employees with salary > 100000
db.Staff.deleteMany({salary: {$gt: 100000}})
OUTPUT: find(): Shows 10 documents. Projection: empid+designation only. sort({salary:-1}): Neha(120000) >

Arjun(90000) > Ravi(70000)... $or query returns Anita, Ravi, Arjun, Neha. updateMany matched:2, modified:1.

deleteMany deletedCount:1.

Q2: Student Collection
INSERT:

db.Student.insertMany([
{RollNo:1, Name:'Aman', Class:'BSc', TotalMarks:320},
{RollNo:2, Name:'Riya', Class:'MSc', TotalMarks:420},
{RollNo:3, Name:'Karan', Class:'BSc', TotalMarks:280},
{RollNo:4, Name:'Sneha', Class:'MSc', TotalMarks:450},
{RollNo:5, Name:'Rohit', Class:'BCom', TotalMarks:150},
{RollNo:6, Name:'Priya', Class:'MSc', TotalMarks:410},
{RollNo:7, Name:'Arjun', Class:'BSc', TotalMarks:390},
{RollNo:8, Name:'Nisha', Class:'MSc', TotalMarks:470},
{RollNo:9, Name:'Vikas', Class:'BCom', TotalMarks:180},
{RollNo:10, Name:'Simran', Class:'BSc', TotalMarks:300}
])
QUERIES:

// Display all documents
db.Student.find()
// Sort in descending order of TotalMarks
db.Student.find().sort({TotalMarks: -1})
// Display MSc students OR TotalMarks > 400
db.Student.find({
$or: [
{Class: 'MSc'},
{TotalMarks: {$gt: 400}}
]
})
// Remove students with TotalMarks < 200
db.Student.deleteMany({TotalMarks: {$lt: 200}})
OUTPUT: sort({TotalMarks:-1}): Nisha(470) > Sneha(450) > Riya(420)... $or returns MSc students +

TotalMarks>400. deleteMany({TotalMarks:{$lt:200}}) deletedCount:2.

Q3: MapReduce – Total Sales by Product
CODE:

// Insert Sales data
db.Sales.insertMany([
{_id:1, product:'apple', amount:100},
{_id:2, product:'banana', amount:150},
{_id:3, product:'apple', amount:200},
{_id:4, product:'oranges', amount:100},
{_id:5, product:'banana', amount:350},
{_id:6, product:'oranges', amount:200}
])

// Map Function
var mapFunction = function() {
