# Exploratory Data Analysis with Python and Pandas

This project demonstrates a complete **Exploratory Data Analysis (EDA)** workflow using Python. The analysis is performed on a supermarket sales dataset to explore customer purchasing behavior, sales trends, and relationships between variables.

The project uses Python data analysis and visualization libraries such as **Pandas**, **NumPy**, **Seaborn**, and **Matplotlib** within a Jupyter Notebook environment to perform statistical exploration, data cleaning, and visualization.

The primary goal of this project is to showcase how EDA can be used to understand dataset structure, detect patterns, handle data quality issues, and derive insights before building machine learning models.

## Dataset

File: `supermarket_sales.csv`

The dataset contains transactional data from a supermarket, including customer demographics, purchase details, payment methods, and sales information.

Example Features:

| Column                  | Description                      |
| ----------------------- | -------------------------------- |
| Invoice ID              | Unique transaction identifier    |
| Branch                  | Supermarket branch (A, B, C)     |
| City                    | City where the branch is located |
| Customer type           | Member or normal customer        |
| Gender                  | Customer gender                  |
| Product line            | Product category                 |
| Unit price              | Price per unit                   |
| Quantity                | Number of items purchased        |
| Tax 5%                  | Tax amount                       |
| Total                   | Total purchase cost              |
| Date                    | Date of transaction              |
| Time                    | Time of transaction              |
| Payment                 | Payment method                   |
| COGS                    | Cost of goods sold               |
| Gross margin percentage | Margin percentage                |
| Gross income            | Income from transaction          |
| Rating                  | Customer rating                  |

## Technologies and Libraries

The following Python libraries are used in this project:
- **Python**
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Calmap** – Calendar heatmap visualization
- **Jupyter Notebook** – Interactive analysis environment

## Exploratory Data Analysis Workflow

The EDA process in this project is organized into several analytical tasks.

### Task 1 — Initial Data Exploration
The first step involves understanding the dataset structure and preparing the environment.

#### Key Steps:
- Import required libraries: Pandas, NumPy, Seaborn, Matplotlib
- Load dataset using Pandas
- Inspect dataset structure
- Convert `Date` column to datetime
- Set `Date` as the index for time series analysis

#### Operations Performed
```
df = pd.read_csv("supermarket_sales.csv")
df.head()
df.columns
df.dtypes
```
This step helps identify:
- Column types
- Data distribution
- Potential data quality issues

### Task 2 — Univariate Analysis
Univariate analysis examines the distribution of individual variables.

#### Continuous Variables
The distribution of **customer ratings** is visualized using Seaborn:
- Histogram
- Kernel Density Estimate (KDE)
- Mean indicator
- 25th and 75th percentile indicators

Example:
```
sns.histplot(df['Rating'], kde=True)
```
#### Distribution of Numeric Features

Using Pandas:
```
df.hist(figsize=(10,10))
```
This helps analyze:
- Spread of numerical variables
- Potential outliers
- Distribution patterns

#### Categorical Variables
Frequency distributions are visualized using **count plots**.

Examples:
- Branch distribution
- Payment method distribution
```
sns.countplot(x=df['Branch'])
sns.countplot(x=df['Payment'])
```

### Task 3 — Bivariate Analysis
Bivariate analysis explores relationships between two variables.

#### Scatter and Regression Analysis

Relationship between **customer rating** and **gross income**:
```
sns.regplot(x=df['Rating'], y=df['gross income'])
```
This helps determine whether higher ratings correlate with higher spending.

#### Sales Comparison by Branch

Boxplots are used to analyze differences in **gross income** across branches.
```
sns.boxplot(x=df['Branch'], y=df['gross income'])
```
#### Sales Comparison by Gender

Boxplots help compare **spending patterns** between genders.
```
sns.boxplot(x=df['Gender'], y=df['gross income'])
```
#### Time Series Analysis

Daily average gross income is analyzed using a line plot.
```
df.groupby(df.index).mean()
```
Visualization:
```
sns.lineplot(x=date, y=gross_income)
```
This helps identify **temporal sales trends** over the three-month period.

#### Pairwise Relationship Visualization

A **pairplot** is generated to visualize relationships between all numeric variables.
```
sns.pairplot(df)
```
This provides a quick overview of correlations and distributions.

### Task 4 — Handling Data Quality Issues
Data quality is critical for reliable analysis.

#### Detect Duplicate Rows
```
df.duplicated().sum()
```
Remove duplicates:
```
df.drop_duplicates(inplace=True)
```
#### Detect Missing Values
```
df.isna().sum()
```
Missing values are visualized using a heatmap:
```
sns.heatmap(df.isnull())
```
#### Handling Missing Data

Numeric values are replaced with column means:
```
df.fillna(df.mean(), inplace=True)
```
Categorical values are replaced using the mode:
```
df.fillna(df.mode().iloc[0], inplace=True)
```

### Task 5 — Correlation Analysis

Correlation analysis identifies relationships between numerical variables.

#### Pairwise Correlation
Using NumPy:
```
np.corrcoef(df['gross income'], df['Rating'])
```
#### Correlation Matrix
Using Pandas:
```
df.corr()
```
Rounded correlation matrix:
```
np.round(df.corr(), 2)
```
#### Heatmap Visualization
The correlation matrix is visualized using a **Seaborn heatmap**.
```
sns.heatmap(df.corr(), annot=True)
```
This visualization helps identify:
- Strong positive correlations
- Weak relationships
- Potential predictive features

---
