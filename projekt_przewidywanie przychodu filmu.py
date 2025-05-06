import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Wczytanie danych z IMDb ===
imdb_df = pd.read_csv("imdb_data.csv")

# === 2. Funkcja do pobierania Google Trends ===
def fetch_trends(titles, batch_size=5, cache_file="trends_data.parquet"):
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    pytrends = TrendReq(hl='en-US', tz=360)
    trends_data = []

    for i in range(0, len(titles), batch_size):
        batch = list(titles[i:i+batch_size])
        try:
            pytrends.build_payload(batch, timeframe='today 12-m')
            df = pytrends.interest_over_time()
            if not df.empty:
                df = df.drop(columns='isPartial')
                avg_scores = df.mean().reset_index()
                avg_scores.columns = ['title', 'trend_score']
                trends_data.append(avg_scores)
        except Exception as e:
            print(f"Błąd przy batchu {batch}: {e}")
        time.sleep(1.5)

    result = pd.concat(trends_data, ignore_index=True)
    result.to_parquet(cache_file)
    return result

# === 3. Pobieranie danych Google Trends ===
google_trends_df = fetch_trends(imdb_df['primaryTitle'].unique())
df = pd.merge(imdb_df, google_trends_df, left_on='primaryTitle', right_on='title', how='left')

# === 4. Czyszczenie danych ===
df['trend_score'] = df['trend_score'].fillna(0)

# Usunięcie outlierów
iso = IsolationForest(contamination=0.05, random_state=42)
mask = iso.fit_predict(df[['numVotes']])
df = df[mask == 1]

# Etykietowanie popularności
threshold = df['numVotes'].quantile(0.75)
df['popular'] = (df['numVotes'] >= threshold).astype(int)

# === 5. Przygotowanie danych do modelu ===
features = ['genres', 'runtimeMinutes', 'averageRating', 'trend_score']
target = 'popular'

X = df[features]
y = df[target]

# One-hot encoding + skalowanie
numeric_features = ['runtimeMinutes', 'averageRating', 'trend_score']
categorical_features = ['genres']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# === 6. Podział na zbiory ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 7. Budowa i trening modelu ===
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# === 8. Wyniki ===
print("\n=== Raport klasyfikacji ===")
print(classification_report(y_test, y_pred))

# === 9. Wykres ważności cech ===
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_df, x='Importance', y='Feature', palette='viridis')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

# Wyciągnięcie nazw cech
rf_model = clf.named_steps['classifier']
ohe = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_names = ohe.get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_names)

# Wykres
plot_feature_importance(rf_model, all_features)
