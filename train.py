import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib


def load_features_from_feature_store():
	"""Try to load features from Hopsworks. If hopsworks isn't available, raise an exception."""
	try:
		import hopsworks
		project = hopsworks.login()
		fs = project.get_feature_store()
		# Try both versions (2 then 1) to be resilient to feature group version changes
		for v in (2, 1):
			try:
				fg = fs.get_feature_group("karachi_air_quality_features", version=v)
				print(f"Loaded feature group 'karachi_air_quality_features' version {v} from Feature Store.")
				return fg.read()
			except Exception:
				continue
		raise RuntimeError("Could not find 'karachi_air_quality_features' in the Feature Store (versions tried: 2,1).")
	except Exception as e:
		raise


def load_features_df():
	"""Load features dataframe from Feature Store if possible, otherwise fallback to local CSV `features_data.csv`."""
	# First try feature store
	try:
		df = load_features_from_feature_store()
		return df
	except Exception as e:
		print(f"Warning: could not load from Feature Store ({e}). Falling back to local 'features_data.csv'.")
		local_path = Path(__file__).parent / "features_data.csv"
		# If not next to this file, try repo root
		if not local_path.exists():
			local_path = Path.cwd() / "features_data.csv"
		if not local_path.exists():
			raise FileNotFoundError("features_data.csv not found locally. Please run feature engineering or provide data.")
		df = pd.read_csv(local_path)
		print(f"Loaded features from {local_path}.")
		return df


def main():
	df = load_features_df()

	# Save a copy locally so other apps can use it
	df.to_csv("features_data.csv", index=False)

	# Prepare data
	if "aqi" not in df.columns:
		raise RuntimeError("Input data must contain 'aqi' column as target.")

	X = df.drop(columns=[c for c in ("aqi", "timestamp") if c in df.columns])
	y = df["aqi"]

	# Remember the feature column order so the API and frontends can reconstruct inputs
	feature_columns = X.columns.tolist()

	# Save feature column list
	with open("feature_columns.json", "w") as f:
		json.dump(feature_columns, f)

	# Train/test split and model
	X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.2, random_state=42)
	model = RandomForestRegressor(n_estimators=100, random_state=42)
	model.fit(X_train, y_train)

	# Save model
	joblib.dump(model, "model.pkl")
	print("Saved trained model to 'model.pkl'.")

	# Metrics
	y_pred = model.predict(X_test)
	print("MAE:", mean_absolute_error(y_test, y_pred))
	print("accuracy:", model.score(X_test, y_test))
	print("training accuracy:", model.score(X_train, y_train))


if __name__ == "__main__":
	main()