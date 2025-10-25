import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

features_fg = fs.get_feature_group("karachi_air_quality_features", version=1)
df = features_fg.read()

df.to_csv("features_data.csv", index=False)

X = df.drop(columns=["aqi", "timestamp"])
y = df["aqi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("accuracy:", model.score(X_test, y_test))
print("training accuracy:", model.score(X_train, y_train))