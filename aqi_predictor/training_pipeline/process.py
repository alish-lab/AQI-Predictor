import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from aqi_predictor.feature_pipeline.api_call import get_hourly_dataframe


def create_sequences(data: np.ndarray, seq_length: int):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def process_air_quality_data(data: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Process the air quality data and return a DataFrame.

    Parameters:
    data (pd.DataFrame | None): The air quality DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing the processed air quality data.
    """
    if data is None:
        data = get_hourly_dataframe()

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Expected a DataFrame for air quality data.")

    processed_df = data.copy()
    processed_df["time"] = pd.to_datetime(processed_df["time"])
    processed_df.replace({None: np.nan}, inplace=True)

    return processed_df


processed_df = process_air_quality_data()

# Candidate features to include in sequences from the API output.
# We keep `us_aqi` as the target while using all air-quality predictors.
candidate_features = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "sulphur_dioxide",
    "ozone",
    "nitrogen_dioxide",
    "us_aqi",
]

# Use only the features that actually exist in the DataFrame
available_features = [c for c in candidate_features if c in processed_df.columns]
if "us_aqi" not in available_features:
    raise ValueError("Required column 'us_aqi' not found in data")

# Coerce selected features to numeric and drop rows with missing values in those columns
processed_df[available_features] = processed_df[available_features].apply(
    lambda col: pd.to_numeric(col, errors="coerce")
)
processed_df = processed_df.dropna(subset=available_features).reset_index(drop=True)

values = processed_df[available_features].astype(float).values
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

seq_length = 10
X, y_full = create_sequences(scaled_values, seq_length)
# If multivariate, extract the `us_aqi` column from the sequence targets as the
# scalar prediction target; otherwise use the single-column y from create_sequences.
if scaled_values.ndim == 1:
    y = y_full
else:
    us_idx = available_features.index("us_aqi")
    # y_full shape: (n_samples, n_features) -> select us_aqi column
    y = y_full[:, us_idx]

trainX, testX, trainY, testY = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

train_scaled = trainX
test_scaled = testX
train_labels = trainY
test_labels = testY

# Export the list of feature column names used to build sequences. This list is
# used by downstream scripts (e.g. SHAP plotting) to create human-readable names
# for flattened sequence features.
feature_columns = available_features





