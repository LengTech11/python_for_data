# Python for Data: A Complete Self-Study Course
### NumPy · Pandas · Matplotlib

A hands-on, project-based course you can work through on your own. Each module has: concepts, code examples, exercises, and a mini-project. No prior data-science experience assumed — basic Python (variables, loops, functions) is enough to start.

**How to use this course**
1. Set up your environment (Module 0).
2. Work through modules in order — each builds on the last.
3. Type the code yourself; don't copy-paste. Run everything.
4. Do every exercise before checking a solution online.
5. Build the capstone project at the end to tie it all together.

**Estimated time:** 4–6 weeks at 1–2 hours/day.

---

## Table of Contents

- Module 0: Environment Setup
- Module 1: Python Refresher for Data Work
- Module 2: NumPy Fundamentals
- Module 3: NumPy — Intermediate & Advanced
- Module 4: Pandas — Series & DataFrames
- Module 5: Pandas — Cleaning & Transforming Data
- Module 6: Pandas — Grouping, Merging, Time Series
- Module 7: Matplotlib Fundamentals
- Module 8: Matplotlib — Advanced Plots & Styling
- Module 9: Putting It Together — EDA Workflow
- Module 10: Capstone Project
- Appendix: Cheat Sheets & Further Resources

---

## Module 0: Environment Setup

### Option A: Anaconda (recommended for beginners)
1. Download Anaconda from anaconda.com and install it.
2. Open "Anaconda Navigator" or run `jupyter notebook` / `jupyter lab` from a terminal.
3. Create a new notebook and run:
   ```python
   import numpy, pandas, matplotlib
   print(numpy.__version__, pandas.__version__, matplotlib.__version__)
   ```

### Option B: Plain Python + pip
```bash
python -m venv data-env
source data-env/bin/activate      # Windows: data-env\Scripts\activate
pip install numpy pandas matplotlib jupyterlab
jupyter lab
```

### Option C: No install — cloud notebooks
Use Google Colab (colab.research.google.com) — free, nothing to install, comes with numpy/pandas/matplotlib pre-installed.

### Folder structure for this course
```
python-data-course/
  data/            # datasets you download or create
  notebooks/       # one notebook per module
  scripts/         # optional .py versions
```

**Checkpoint:** You should be able to open a notebook and successfully import all three libraries with no errors.

---

## Module 1: Python Refresher for Data Work

Skip this if you're already comfortable with Python. Otherwise, make sure you can do the following before continuing.

### 1.1 Core building blocks
```python
# Variables & types
x = 5
y = 3.14
name = "data"
is_valid = True

# Lists, tuples, dicts, sets
lst = [1, 2, 3]
tup = (1, 2, 3)
d = {"a": 1, "b": 2}
s = {1, 2, 3}

# List comprehensions (you will use these CONSTANTLY)
squares = [n**2 for n in range(10)]
evens = [n for n in range(20) if n % 2 == 0]
```

### 1.2 Functions
```python
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Lambda functions — used a lot with pandas
square = lambda x: x ** 2
```

### 1.3 File I/O basics
```python
with open("data/sample.txt", "r") as f:
    content = f.read()
```

### 1.4 Exercises
1. Write a list comprehension that returns the squares of all odd numbers from 1–50.
2. Write a function `celsius_to_fahrenheit(c)` and test it on `[0, 20, 37, 100]` using a list comprehension.
3. Create a dictionary mapping 5 country names to their capitals, then print each as `"<capital> is the capital of <country>"`.

---

## Module 2: NumPy Fundamentals

NumPy is the foundation of the Python data stack — pandas and most ML libraries are built on top of it.

### 2.1 Why NumPy?
Python lists are slow for numeric work and don't support vectorized math. NumPy arrays (`ndarray`) are fixed-type, contiguous in memory, and support fast vectorized operations.

```python
import numpy as np

py_list = list(range(1_000_000))
np_array = np.arange(1_000_000)

# %timeit [x*2 for x in py_list]      # slow
# %timeit np_array * 2                # much faster
```

### 2.2 Creating arrays
```python
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])          # 2D

np.zeros((3, 4))
np.ones((2, 2))
np.full((2, 2), 7)
np.eye(3)                                      # identity matrix
np.arange(0, 10, 2)                            # like range()
np.linspace(0, 1, 5)                           # 5 evenly spaced points

np.random.seed(42)
np.random.rand(3, 3)                           # uniform [0,1)
np.random.randn(3, 3)                          # standard normal
np.random.randint(0, 10, size=(3, 3))
```

### 2.3 Array attributes
```python
a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape      # (2, 3)
a.ndim       # 2
a.size       # 6
a.dtype      # int64
a.reshape(3, 2)
a.flatten()
```

### 2.4 Indexing & slicing
```python
arr = np.array([10, 20, 30, 40, 50])
arr[0]        # 10
arr[-1]       # 50
arr[1:4]      # [20, 30, 40]
arr[::-1]     # reversed

mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mat[1, 2]     # 6
mat[:, 0]     # first column: [1, 4, 7]
mat[0, :]     # first row: [1, 2, 3]
mat[0:2, 1:3] # sub-matrix
```

### 2.5 Boolean masking & fancy indexing
```python
arr = np.array([1, -2, 3, -4, 5])
mask = arr > 0
arr[mask]            # [1, 3, 5]
arr[arr > 0] = 0     # conditional assignment

idx = np.array([0, 2, 4])
arr[idx]             # fancy indexing
```

### 2.6 Vectorized operations
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b, a - b, a * b, a / b, a ** 2
np.sqrt(a)
np.exp(a)
np.log(a)
a.sum(), a.mean(), a.std(), a.min(), a.max()
a.argmax(), a.argmin()
```

### 2.7 Exercises
1. Create a 5×5 array of random integers between 1 and 100. Find its mean, max, and standard deviation.
2. Create an array of the numbers 1–20. Replace every even number with 0 using boolean masking.
3. Create two 1D arrays of length 10 and compute their dot product manually with a loop, then verify with `np.dot`.
4. Create a 4×4 identity matrix and multiply it by 5.

---

## Module 3: NumPy — Intermediate & Advanced

### 3.1 Broadcasting
NumPy lets arrays of different shapes be combined without explicit loops, following broadcasting rules.
```python
a = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2,3)
b = np.array([10, 20, 30])              # shape (3,)
a + b   # b is broadcast across each row

col = np.array([[1], [2]])              # shape (2,1)
a + col  # broadcast across each column
```

### 3.2 Axis-based aggregation
```python
mat = np.array([[1, 2, 3], [4, 5, 6]])
mat.sum(axis=0)   # sum down each column -> [5, 7, 9]
mat.sum(axis=1)   # sum across each row  -> [6, 15]
mat.mean(axis=0)
```

### 3.3 Stacking, splitting, concatenation
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.concatenate([a, b])
np.vstack([a, b])
np.hstack([a, b])
np.split(np.arange(10), 5)
```

### 3.4 Linear algebra basics
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A @ B                 # matrix multiplication
np.linalg.inv(A)      # inverse
np.linalg.det(A)      # determinant
np.linalg.eig(A)      # eigenvalues/eigenvectors
A.T                    # transpose
```

### 3.5 Sorting and searching
```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
np.sort(arr)
np.argsort(arr)          # indices that would sort the array
np.where(arr > 3)        # indices where condition true
np.unique(arr)
```

### 3.6 Working with NaN / missing values
```python
arr = np.array([1, 2, np.nan, 4])
np.isnan(arr)
np.nanmean(arr)      # ignores NaN
np.nansum(arr)
```

### 3.7 Vectorization vs loops (why it matters)
```python
# Slow: Python loop
result = []
for x in range(1000000):
    result.append(x ** 2)

# Fast: vectorized
result = np.arange(1_000_000) ** 2
```
Rule of thumb: if you're writing a `for` loop over a NumPy array to do math, there's almost always a vectorized way to do it faster.

### 3.8 Exercises
1. Given a 3×3 matrix, compute its row sums, column sums, and total sum without using `axis` twice redundantly.
2. Create two arrays of shape (3,1) and (1,4) and use broadcasting to produce a (3,4) result.
3. Given `scores = np.array([55, 72, 90, 61, 45, 88])`, use `np.where` to label each as "pass" (>=60) or "fail".
4. Create a 5x5 matrix, replace all NaN values (insert a few manually) with the column mean.

---

## Module 4: Pandas — Series & DataFrames

Pandas is built on NumPy and adds labeled, tabular data structures — the workhorse for real-world data analysis.

### 4.1 Series (1D labeled array)
```python
import pandas as pd

s = pd.Series([10, 20, 30], index=["a", "b", "c"])
s["a"]          # 10
s.values        # underlying numpy array
s.index

s2 = pd.Series({"x": 1, "y": 2, "z": 3})
```

### 4.2 DataFrame (2D labeled table)
```python
data = {
    "name": ["Alice", "Bob", "Carol", "Dave"],
    "age": [25, 32, 28, 41],
    "city": ["NYC", "LA", "NYC", "SF"]
}
df = pd.DataFrame(data)
df.head()
df.tail(2)
df.shape
df.columns
df.dtypes
df.info()
df.describe()          # summary stats for numeric columns
```

### 4.3 Reading & writing data
```python
df = pd.read_csv("data/sample.csv")
df = pd.read_excel("data/sample.xlsx")
df = pd.read_json("data/sample.json")

df.to_csv("data/output.csv", index=False)
df.to_excel("data/output.xlsx", index=False)
```

### 4.4 Selecting data
```python
df["age"]                     # single column -> Series
df[["name", "age"]]           # multiple columns -> DataFrame

df.loc[0]                     # row by label
df.loc[0:2, "name"]           # label-based slicing
df.iloc[0]                    # row by position
df.iloc[0:2, 0:2]             # position-based slicing

df[df["age"] > 28]            # boolean filtering
df[(df["age"] > 25) & (df["city"] == "NYC")]   # multiple conditions
```

### 4.5 Adding, modifying, dropping columns
```python
df["age_in_5"] = df["age"] + 5
df["is_adult"] = df["age"] >= 18
df.drop("age_in_5", axis=1, inplace=True)
df.rename(columns={"name": "full_name"}, inplace=True)
```

### 4.6 Sorting
```python
df.sort_values("age")
df.sort_values("age", ascending=False)
df.sort_values(["city", "age"])
```

### 4.7 Exercises
1. Load any CSV (or create one manually with a dict of 10 rows) and print its shape, columns, and dtypes.
2. Select all rows where a numeric column is above its own mean.
3. Add a new column that's a transformation of an existing one (e.g., a discount price).
4. Sort a DataFrame by two columns, one ascending and one descending.

---

## Module 5: Pandas — Cleaning & Transforming Data

Real-world data is messy. This module covers the tools you'll use most.

### 5.1 Handling missing data
```python
df.isnull()
df.isnull().sum()                  # missing values per column
df.dropna()                        # drop rows with any NaN
df.dropna(subset=["age"])          # drop rows missing this column
df.fillna(0)
df["age"].fillna(df["age"].mean(), inplace=True)
df.fillna(method="ffill")          # forward fill
```

### 5.2 Duplicates
```python
df.duplicated()
df.drop_duplicates()
df.drop_duplicates(subset=["name"], keep="first")
```

### 5.3 Data type conversion
```python
df["age"] = df["age"].astype(int)
df["date"] = pd.to_datetime(df["date"])
df["category"] = df["category"].astype("category")
```

### 5.4 String operations (the `.str` accessor)
```python
df["name"] = df["name"].str.lower()
df["name"] = df["name"].str.strip()
df["initials"] = df["name"].str[0]
df["contains_a"] = df["name"].str.contains("a")
df["parts"] = df["name"].str.split(" ")
```

### 5.5 Applying custom functions
```python
df["age_group"] = df["age"].apply(lambda x: "adult" if x >= 18 else "minor")

def categorize(row):
    if row["age"] < 18:
        return "minor"
    elif row["age"] < 65:
        return "adult"
    else:
        return "senior"

df["age_group"] = df.apply(categorize, axis=1)   # row-wise
```

### 5.6 map, replace, and binning
```python
df["city_code"] = df["city"].map({"NYC": 1, "LA": 2, "SF": 3})
df["city"] = df["city"].replace({"NYC": "New York"})

df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 100],
                        labels=["minor", "young_adult", "adult", "senior"])
```

### 5.7 Exercises
1. Take a DataFrame with intentionally missing values and demonstrate three different strategies for handling them (drop, fill with mean, forward fill).
2. Clean a "name" column that has inconsistent casing and extra whitespace.
3. Use `apply` to create a new column classifying rows into 3+ custom categories based on multiple conditions.
4. Use `pd.cut` to bin a numeric column into 4 labeled ranges.

---

## Module 6: Pandas — Grouping, Merging, Time Series

### 6.1 GroupBy — split-apply-combine
```python
df.groupby("city")["age"].mean()
df.groupby("city").agg({"age": ["mean", "min", "max"], "name": "count"})
df.groupby(["city", "age_group"]).size()

for city, group in df.groupby("city"):
    print(city, len(group))
```

### 6.2 Pivot tables
```python
pd.pivot_table(df, values="age", index="city", columns="age_group", aggfunc="mean")
df.pivot_table(index="city", values="age", aggfunc=["mean", "count"])
```

### 6.3 Merging & joining
```python
orders = pd.DataFrame({"order_id": [1, 2, 3], "customer_id": [1, 2, 1]})
customers = pd.DataFrame({"customer_id": [1, 2], "name": ["Alice", "Bob"]})

pd.merge(orders, customers, on="customer_id", how="inner")
pd.merge(orders, customers, on="customer_id", how="left")
pd.merge(orders, customers, on="customer_id", how="outer")

pd.concat([df1, df2])                 # stack vertically
pd.concat([df1, df2], axis=1)         # stack horizontally
```

### 6.4 Time series basics
```python
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

df["2023"]                     # all rows from 2023
df["2023-01":"2023-03"]        # date range slicing

df.resample("M").mean()        # monthly aggregation
df.resample("W").sum()         # weekly aggregation

df["day_of_week"] = df.index.day_name()
df["month"] = df.index.month
df["rolling_avg"] = df["value"].rolling(window=7).mean()
```

### 6.5 Exercises
1. Group a sales-style dataset by category and compute total and average sales per category.
2. Build a pivot table showing average value across two categorical dimensions.
3. Create two related DataFrames (e.g., orders and products) and merge them with an inner join, then a left join — compare row counts.
4. Create a date-indexed DataFrame of daily values over a year and compute a 7-day rolling average.

---

## Module 7: Matplotlib Fundamentals

### 7.1 The basic plot
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("x squared")
plt.show()
```

### 7.2 The object-oriented interface (preferred for real work)
```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label="y = x^2", color="steelblue", linewidth=2)
ax.set_title("Squares")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
```

### 7.3 Common chart types
```python
fig, ax = plt.subplots()
ax.bar(["A", "B", "C"], [10, 20, 15])          # bar chart
ax.barh(["A", "B", "C"], [10, 20, 15])         # horizontal bar

fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.7)                     # scatter plot

fig, ax = plt.subplots()
ax.hist(np.random.randn(1000), bins=30)         # histogram

fig, ax = plt.subplots()
ax.pie([30, 20, 50], labels=["A", "B", "C"], autopct="%1.1f%%")  # pie chart

fig, ax = plt.subplots()
ax.boxplot([np.random.randn(100), np.random.randn(100) + 1])    # boxplot
```

### 7.4 Multiple subplots
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y)
axes[0, 1].bar(["A", "B"], [1, 2])
axes[1, 0].scatter(x, y)
axes[1, 1].hist(np.random.randn(500))
plt.tight_layout()
plt.show()
```

### 7.5 Saving figures
```python
fig.savefig("output.png", dpi=300, bbox_inches="tight")
```

### 7.6 Exercises
1. Plot a sine and cosine wave on the same axes with a legend, using `np.linspace` and `np.sin`/`np.cos`.
2. Create a bar chart comparing 5 categories, with the bars colored differently.
3. Create a 2x2 grid of subplots showing 4 different chart types on related data.
4. Save one of your figures as a PNG at 300 DPI.

---

## Module 8: Matplotlib — Advanced Plots & Styling

### 8.1 Styling
```python
plt.style.use("seaborn-v0_8")     # or "ggplot", "fivethirtyeight", etc.
print(plt.style.available)        # list all available styles
```

### 8.2 Customizing appearance
```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, color="#e63946", linestyle="--", marker="o", markersize=8)
ax.set_xlim(0, 6)
ax.set_ylim(0, 30)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

### 8.3 Annotations
```python
ax.annotate("Peak", xy=(5, 25), xytext=(3, 27),
            arrowprops=dict(facecolor="black", shrink=0.05))
ax.text(2, 10, "Note here", fontsize=10)
```

### 8.4 Plotting directly from pandas
```python
df["age"].plot(kind="hist", bins=20)
df.plot(x="date", y="value", kind="line")
df.groupby("city")["age"].mean().plot(kind="bar")
df.plot.scatter(x="age", y="income")
```

### 8.5 Combining with seaborn (optional but common)
```python
# pip install seaborn
import seaborn as sns

sns.histplot(df["age"], kde=True)
sns.boxplot(x="city", y="age", data=df)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
sns.pairplot(df)
```

### 8.6 Exercises
1. Recreate one of your Module 7 charts using a matplotlib style sheet and clean, spine-free axes.
2. Add an annotation pointing to the maximum value in a line chart.
3. Plot a histogram and a bar chart directly from a pandas DataFrame/Series without calling `plt` directly.
4. (Optional) Install seaborn and create a heatmap of correlations for a numeric dataset.

---

## Module 9: Putting It Together — EDA Workflow

Exploratory Data Analysis (EDA) is the standard workflow professionals use. Practice it end-to-end on any dataset (try Titanic, Iris, or a dataset from kaggle.com/datasets).

### 9.1 The standard EDA checklist
1. **Load & inspect**: `df.shape`, `df.head()`, `df.info()`, `df.describe()`
2. **Check for missing data**: `df.isnull().sum()`
3. **Check for duplicates**: `df.duplicated().sum()`
4. **Understand each column**: data type, unique values, distribution
5. **Univariate analysis**: histograms/bar charts for individual columns
6. **Bivariate analysis**: scatter plots, group comparisons, correlations
7. **Handle outliers & missing values** based on what you found
8. **Summarize findings** in a few sentences

### 9.2 Worked example structure
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/dataset.csv")

# 1. Structure
print(df.shape)
df.info()
df.describe()

# 2. Missing values
missing = df.isnull().sum()
print(missing[missing > 0])

# 3. Distributions
df.hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

# 4. Correlations
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
plt.colorbar(im)
plt.show()

# 5. Group comparisons
df.groupby("category")["target"].mean().plot(kind="bar")
plt.show()
```

### 9.3 Exercise
Pick any public CSV dataset and run the full checklist above, writing 5–10 bullet points of findings (e.g., "Column X has 12% missing values", "Feature Y is right-skewed", "Category A has a notably higher average than B").

---

## Module 10: Capstone Project

Choose ONE of these (or bring your own dataset/question):

### Option A: Sales Analysis
Using a sales dataset (date, product, category, quantity, price, region):
- Clean the data (missing values, duplicates, correct types)
- Compute total revenue by month, region, and category
- Find the top 5 best-selling products
- Plot monthly revenue trend with a rolling average
- Plot a bar chart of revenue by region
- Write a short summary of your findings

### Option B: Weather Data Analysis
Using a daily weather dataset (date, temp, precipitation, city):
- Convert date column properly, set as index
- Compute monthly average temperature per city
- Identify the hottest and coldest months
- Plot temperature trends for multiple cities on one chart
- Compute correlation between temperature and precipitation

### Option C: Personal Dataset
Use data you actually care about — your own expenses, workout logs, or a hobby (e.g., BookMeBus-style transit/booking data if relevant to your own work). Apply the full EDA + visualization workflow from Module 9.

### Deliverable
A single notebook (or `.py` script) containing:
1. Data loading & cleaning
2. At least 3 pandas transformations (groupby, merge, or pivot)
3. At least 4 different chart types
4. A written summary (5–10 bullet points) of what the data shows

---

## Appendix: Cheat Sheets & Further Resources

### Quick reference: NumPy
| Task | Code |
|---|---|
| Create array | `np.array([1,2,3])` |
| Shape | `arr.shape` |
| Reshape | `arr.reshape(r, c)` |
| Slice | `arr[start:stop:step]` |
| Boolean filter | `arr[arr > 0]` |
| Sum/Mean | `arr.sum()`, `arr.mean()` |
| Matrix multiply | `A @ B` |

### Quick reference: Pandas
| Task | Code |
|---|---|
| Read CSV | `pd.read_csv(path)` |
| Filter rows | `df[df["col"] > x]` |
| Select columns | `df[["a", "b"]]` |
| Group & aggregate | `df.groupby("col").agg(...)` |
| Merge | `pd.merge(df1, df2, on="key")` |
| Missing values | `df.isnull().sum()`, `df.fillna()` |
| Apply function | `df["col"].apply(func)` |

### Quick reference: Matplotlib
| Task | Code |
|---|---|
| Line plot | `ax.plot(x, y)` |
| Bar chart | `ax.bar(labels, values)` |
| Scatter | `ax.scatter(x, y)` |
| Histogram | `ax.hist(data, bins=n)` |
| Save figure | `fig.savefig("name.png", dpi=300)` |

### Recommended free resources
- Official docs: numpy.org/doc, pandas.pydata.org/docs, matplotlib.org/stable
- Practice datasets: kaggle.com/datasets, UCI Machine Learning Repository
- Practice problems: kaggle.com "Learn" micro-courses, w3resource.com pandas/numpy exercises

### Suggested pace
| Week | Focus |
|---|---|
| 1 | Modules 0–2 (setup, Python refresher, NumPy basics) |
| 2 | Module 3 (NumPy advanced) + Module 4 (Pandas basics) |
| 3 | Modules 5–6 (Pandas cleaning, grouping, merging) |
| 4 | Modules 7–8 (Matplotlib) |
| 5 | Module 9 (EDA workflow practice on 2–3 datasets) |
| 6 | Module 10 (Capstone project) |

---

*End of course. Revisit modules as reference material while working on real projects — that's the fastest way to make it stick.*
