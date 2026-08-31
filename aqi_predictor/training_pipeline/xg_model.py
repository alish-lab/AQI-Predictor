import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from aqi_predictor.training_pipeline.process import (
    train_scaled,
    test_scaled,
    train_labels,
    test_labels,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_X = train_scaled.reshape(train_scaled.shape[0], -1)
test_X = test_scaled.reshape(test_scaled.shape[0], -1)
train_y = train_labels.ravel()
test_y = test_labels.ravel()

dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'seed': 42,
}

evals = [(dtrain, 'train'), (dtest, 'eval')]

booster = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=20,
)

predictions = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))

mae = mean_absolute_error(test_y, predictions)
mse = mean_squared_error(test_y, predictions)
rmse = np.sqrt(mse)
accuracy = 1 - (mae / np.mean(test_y))  # Simple accuracy metric for regression

print(f"Best iteration: {booster.best_iteration}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(test_y, label='Actual')
plt.plot(predictions, label='Predicted', linestyle='--')
plt.title('XGBoost Predictions vs Actual')
plt.xlabel('Sample')
plt.ylabel('Scaled us_aqi')
plt.legend()
plt.show()
