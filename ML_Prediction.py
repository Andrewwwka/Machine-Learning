import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")


# --- 1. Data Loading and Initial Inspection ---

df = pd.read_csv('boxoffice.csv',
                 encoding='latin-1')

print("--- First 5 Rows of the DataFrame ---")
print(df.head())

print("\n--- Shape of the DataFrame (Rows, Columns) ---")
print(df.shape)

print("\n--- DataFrame Info ---")
df.info()

print("\n--- Descriptive Statistics (Transposed) ---")
print(df.describe().T)


# --- 2. Data Cleaning and Preprocessing ---

to_remove = ['world_revenue','opening_revenue']
df.drop(to_remove, axis=1, inplace=True)

print("\n--- Percentage of Null Values per Column ---")
print(df.isnull().sum() * 100 / df.shape[0])

# Dropping 'budget' due to high null values
df.drop('budget', axis=1, inplace=True)

# Filling nulls with the mode for categorical columns
for col in ['MPAA','genres']:
    df[col].fillna(df[col].mode()[0], inplace=True) # Used inplace=True for clarity

# Dropping any remaining rows with null values
df.dropna(inplace=True)

print("\n--- Total Number of Null Values after Cleaning ---")
print(df.isnull().sum().sum())

# Cleaning and converting data types for revenue, theaters, and release days
df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1:]
for col in ['domestic_revenue','opening_theaters','release_days']:
    df[col] = df[col].astype(str).str.replace(',','')
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n--- First 5 Rows After Cleaning Numeric Columns ---")
print(df[['domestic_revenue', 'opening_theaters', 'release_days']].head())


# --- 3. Exploratory Data Analysis (EDA) ---

# This section generates plots. plt.show() will display them.
print("\n--- Generating Plots for EDA ---")
print("Plot 1: Count of movies per MPAA rating.")
plt.figure(figsize=(10,5))
sb.countplot(x=df['MPAA']) # Added x= for clarity
plt.show()

print("\n--- Mean Domestic Revenue by MPAA Rating ---")
print(df.groupby('MPAA')['domestic_revenue'].mean())

print("\nPlot 2: Distributions of revenue, theaters, and release days (before log transform).")
plt.subplots(figsize=(15,5))
features_to_plot = ['domestic_revenue','opening_theaters','release_days']
for i, col in enumerate(features_to_plot):
    plt.subplot(1,3,i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

print("\nPlot 3: Boxplots to check for outliers (before log transform).")
plt.subplots(figsize=(15, 5))
for i, col in enumerate(features_to_plot):
    plt.subplot(1, 3, i+1)
    sb.boxplot(y=df[col]) # Added y= for clarity
plt.tight_layout()
plt.show()

# Applying log transformation to handle skewness
for col in features_to_plot:
    # Adding a small constant to avoid log(0) errors if any value is 0
    df[col] = df[col].apply(lambda x: np.log10(x + 1e-6))

print("\nPlot 4: Distributions after applying log transform.")
plt.subplots(figsize=(15, 5))
for i, col in enumerate(features_to_plot):
    plt.subplot(1, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()


# --- 4. Feature Engineering ---

# Vectorizing the 'genres' column
vectorizer = CountVectorizer()
vectorizer.fit(df['genres'])
genre_features = vectorizer.transform(df['genres']).toarray()

# Creating new columns for each genre
genres = vectorizer.get_feature_names_out()
for i, name in enumerate(genres):
    df[name] = genre_features[:,i]

df.drop('genres', axis=1, inplace=True)

# Removing very sparse genre columns (present in < 5% of movies)
removed_count = 0
if 'action' in df.columns and 'western' in df.columns:
    # Assuming 'action' to 'western' covers all your new genre columns
    for col in df.loc[:, 'action':'western']:
        if (df[col] == 0).sum() / len(df) > 0.95:
            removed_count += 1
            df.drop(col, axis=1, inplace=True)

print(f"\nNumber of sparse genre columns removed: {removed_count}")
print(f"Shape of DataFrame after removing sparse genres: {df.shape}")

# Encoding categorical features
for col in ['distributor','MPAA']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Displaying a heatmap of highly correlated features
print("\nPlot 5: Heatmap of feature correlation (> 0.8).")
plt.figure(figsize=(10,8)) # Adjusted size slightly for better fit
sb.heatmap(df.select_dtypes(include=np.number).corr() > 0.8,
           annot=True,
           cbar=False,
           cmap='viridis') # Added a colormap for better visibility
plt.show()


# --- 5. Model Training and Evaluation ---

features = df.drop(['title','domestic_revenue'], axis=1)
target = df['domestic_revenue'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                  test_size=0.1,
                                                  random_state=22)

print("\n--- Data Shapes after Splitting ---")
print(f"Training Features Shape: {X_train.shape}")
print(f"Validation Features Shape: {X_val.shape}")

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

from sklearn.metrics import mean_absolute_error as mae

# Training the model
print("\n--- Training XGBoost Regressor Model ---")
model = XGBRegressor(random_state=42) # Added random_state for reproducibility
model.fit(X_train,Y_train)
print("Model training complete.")

# Making predictions and evaluating the model
train_preds = model.predict(X_train)
print('\n--- Model Performance ---')
print(f'Training MAE (Log Scale) : {mae(Y_train, train_preds):.4f}')

val_preds = model.predict(X_val)
print(f'Validation MAE (Log Scale) : {mae(Y_val, val_preds):.4f}')