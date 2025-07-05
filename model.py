import pandas as pd

df = pd.read_csv('adult 3.csv')
print(df.head())
print(df.info())

df.replace('?', pd.NA, inplace=True)
print(df.isnull().sum())
print(df.describe())
categorical_features = df.select_dtypes(include='object').columns
for col in categorical_features:
    print(df[col].value_counts())

for col in categorical_features:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

print(df.isnull().sum())
df = pd.get_dummies(df, columns=categorical_features, dummy_na=False)
print(df.head())
from sklearn.model_selection import train_test_split

X = df.drop(['income_<=50K', 'income_>50K'], axis=1)
y = df['income_>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


print("Model training complete.")
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse}")

print(f"R-squared (R2) Score: {r2}")
predictions = model.predict(X_test)
print(predictions[:5])