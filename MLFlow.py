import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from skopt import gp_minimize
from skopt.space import Real, Integer

import mlflow
import mlflow.pytorch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# dir_name = 'output_baye_filter_5_n_splits_PM2.5'
# round_num = "t-1"
# os.makedirs(dir_name, exist_ok=True)
# os.makedirs(os.path.join(dir_name, round_num), exist_ok=True)


csv_file_path = r"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\partition_combined_data_upsampled_pm_3H_spline_1degreeM.csv"
df = pd.read_csv(csv_file_path, encoding="utf8")
print(df)
print(f'Dataframe: \n{df.describe()}')
df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d/%m/%Y %H:%M")
print(f"Dataframe to DateTime: \n{df.head(10)}")
df.set_index('DateTime', inplace=True)
# print(f"Dataframe Set Index: \n{df.head(10)}")
df = df.resample('1440T').mean()
print(f"DataFrame Daily: \n{df.head(10)}")


print('Original Data Summary:')
print(df.describe().to_string(), "\n")

input_features = ['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'Season']
target_feature = 'PM2.5'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("PM2.5_Prediction")

# df["PM2.5B"] = df["PM2.5"].shift(1)
# input_features = ['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'Season', 'PM2.5B']
# target_feature = 'PM2.5'
df.dropna(inplace=True)

df = df[input_features + [target_feature]]
df = df.loc[:, ~df.columns.duplicated()]

def compute_zscore(series):
    series = pd.to_numeric(series, errors='coerce')
    std = series.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(0, index=series.index)
    else:
        return (series - series.mean(skipna=True)) / std

df_z = pd.DataFrame({col: compute_zscore(df[col]) for col in df.columns})
filtered_entries = (np.abs(df_z) < 3).all(axis=1)
df = df[filtered_entries]

print("After outlier removal:")
print(df.describe().to_string(), "\n")

X = df[input_features].values
y = df[target_feature].values

print("Data Info:")
print("Max y:", np.max(y))
print("Min y:", np.min(y))
print("Any inf in y:", np.any(np.isinf(y)))
print("Any NaN in y:", np.any(np.isnan(y)))


input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = input_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()


tscv = TimeSeriesSplit(n_splits=5)


def create_dataset(X, y, lookback):
    if len(X) <= lookback:
        raise ValueError("Lookback period longer than dataset")
    X_list, y_list = [], []
    for i in range(len(X) - lookback):
        X_list.append(X[i:i + lookback])
        y_list.append(y[i + lookback])
    X_array = np.array(X_list)
    y_array = np.array(y_list)
    return torch.from_numpy(X_array).float(), torch.from_numpy(y_array).float()


class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, device):
        super(MultivariateLSTM, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            device=device
        ).to(device)
        self.linear = nn.Linear(hidden_size, 1).to(device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        return predictions


class LSTMWrapper:
    def __init__(self):
        self.model = None
        self.input_size = len(input_features)
        self.current_params = None

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y, lookback=4, hidden_size=50, num_layers=2, learning_rate=0.01,
            dropout=0.1, batch_size=64, epochs=200):
        start_time = time.time()
        self.current_params = locals().copy()
        del self.current_params['self']
        del self.current_params['X']
        del self.current_params['y']

        try:
            X_tensor, y_tensor = create_dataset(X, y, lookback)
            X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)

            self.model = MultivariateLSTM(
                self.input_size,
                hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout,
                device=device
            )
            self.model = self.model.to(device)

            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            loss_fn = nn.MSELoss()

            loader = data.DataLoader(
                data.TensorDataset(X_tensor, y_tensor),
                batch_size=batch_size,
                shuffle=True
            )

            for epoch in range(epochs):
                self.model.train()
                epoch_losses = []
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = loss_fn(y_pred.squeeze(), y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_losses.append(loss.item())
                if epoch % 50 == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {np.mean(epoch_losses):.4f}")
            return self

        except Exception as e:
            print(f"Error in fit: {e}")
            return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_tensor, _ = create_dataset(X, np.zeros(len(X)), self.current_params['lookback'])
        X_tensor = X_tensor.to(device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def score(self, X, y):
        """Return negative MSE score for optimization"""
        try:
            predictions = self.predict(X)
            y_seq = y[self.current_params['lookback']:]
            mse = np.mean((predictions.squeeze() - y_seq) ** 2)
            return -mse
        except Exception as e:
            print(f"Error in score: {e}")
            return float('-inf')


def run_optimization(tuning_method='bayes'):
    results_list = []
    start_time = time.time()
    trial_counter = 0

    if tuning_method == 'bayes':
        search_space = [
            Integer(1, 20, name='lookback'),
            Integer(8, 1024, name='hidden_size'),
            Integer(1, 20, name='num_layers'),
            Real(0.001, 0.05, prior='log-uniform', name='learning_rate'),
            # ปรับค่า dropout หากต้องการ (ในที่นี้ comment ไว้)
            Integer(8, 128, name='batch_size'),
            Integer(300, 1500, name='epochs')
        ]

        def objective(params):
            nonlocal trial_counter
            trial_counter += 1

            lookback, hidden_size, num_layers, learning_rate, batch_size, epochs = params
            current_params = {
                'lookback': int(lookback),
                'hidden_size': int(hidden_size),
                'num_layers': int(num_layers),
                'learning_rate': float(learning_rate),
                'batch_size': int(batch_size),
                'epochs': int(epochs),
            }

            try:
                print(f"\nTrial {trial_counter} - Training with params: {current_params}")

                # เริ่มต้นการ Tracking ด้วย MLflow
                with mlflow.start_run(nested=True):
                    # บันทึกพารามิเตอร์ของโมเดล
                    mlflow.log_params(current_params)

                    model = LSTMWrapper()

                    train_size, test_size = 0, 0
                    actual_train, actual_test = [], []
                    train_pred, test_pred = [], []

                    splits = list(tscv.split(X_scaled))
                    previous_train_end = 0

                    for split_idx, (train_index, test_index) in enumerate(splits):
                        print(f"\nProcessing split {split_idx + 1}/{len(splits)}")

                        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                        y_train, y_test = y_scaled[train_index], y_scaled[test_index]

                        previous_train_end = max(train_index) + 1
                        current_epochs = current_params['epochs'] if split_idx == 0 else current_params['epochs'] // 2

                        model.fit(X_train, y_train, **{**current_params, 'epochs': current_epochs})

                        train_predictions = model.predict(X_train).reshape(-1, 1)
                        test_predictions = model.predict(X_test).reshape(-1, 1)

                        train_pred.append(train_predictions)
                        test_pred.append(test_predictions)

                        actual_train.append(y_train[current_params['lookback']:].reshape(-1, 1))
                        actual_test.append(y_test[current_params['lookback']:].reshape(-1, 1))

                        train_size += len(X_train)
                        test_size += len(X_test)

                    train_pred = np.concatenate(train_pred)
                    test_pred = np.concatenate(test_pred)
                    actual_train = np.concatenate(actual_train)
                    actual_test = np.concatenate(actual_test)

                    train_pred_actual = target_scaler.inverse_transform(train_pred)
                    test_pred_actual = target_scaler.inverse_transform(test_pred)
                    actual_train = target_scaler.inverse_transform(actual_train)
                    actual_test = target_scaler.inverse_transform(actual_test)

                    train_rmse = np.sqrt(np.mean((train_pred_actual - actual_train) ** 2))
                    test_rmse = np.sqrt(np.mean((test_pred_actual - actual_test) ** 2))
                    r2 = r2_score(actual_test, test_pred_actual)
                    mae = mean_absolute_error(actual_test, test_pred_actual)

                    # บันทึก Metric ต่าง ๆ ด้วย MLflow
                    mlflow.log_metrics({
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'test_r2': r2,
                        'test_mae': mae
                    })
                    print(f'''
                            'train_rmse': {train_rmse},
                            'test_rmse': {test_rmse},
                            'test_r2': {r2},
                            'test_mae': {mae}
                          ''')

                    # บันทึกโมเดลด้วย MLflow
                    mlflow.pytorch.log_model(model.model, "LSTM_Model")

                    trial_results = {
                        **current_params,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'test_mae': mae,
                        'test_r2': r2,
                        'training_time': time.time() - start_time,
                        'trial_number': trial_counter
                    }
                    results_list.append(trial_results)

                    print(f"Trial {trial_counter} completed in {trial_results['training_time']:.2f} seconds with test RMSE: {test_rmse:.4f}\n")
                    return test_rmse

            except Exception as e:
                print(f"Error in trial {trial_counter}: {e}")
                return float('inf')

        print("Starting Bayesian optimization...")
        result = gp_minimize(
            objective,
            search_space,
            n_calls=100,
            n_random_starts=10,
            random_state=42
        )
        print("Bayesian optimization completed.")

        best_params = {
            'lookback': int(result.x[0]),
            'hidden_size': int(result.x[1]),
            'num_layers': int(result.x[2]),
            'learning_rate': float(result.x[3]),
            'batch_size': int(result.x[4]),
            'epochs': int(result.x[5])
        }

        results_list.append({
            **best_params,
            'train_rmse': None,
            'test_rmse': result.fun,
            'training_time': time.time() - start_time,
            'trial_number': trial_counter
        })

        print(f"Best parameters found: {best_params}")

    # final_csv = os.path.join(dir_name, round_num, f"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\LSTMCV\result\final_optimization_results_{tuning_method}.csv")
    final_results_df = pd.DataFrame(results_list)
    final_results_df.to_csv(fr"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\LSTMCV\result\final_optimization_results_{tuning_method}_original.csv", index=False)

    if len(results_list) > 0:
        best_config = min(results_list, key=lambda x: x['test_rmse'])
        print("\nBest Configuration:")
        for key, value in best_config.items():
            print(f"{key}: {value}")

    return results_list


if __name__ == '__main__':
    tuning_method = 'bayes'
    results_list = run_optimization(tuning_method)

    # output_results = os.path.join(dir_name, round_num, f"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\LSTMCV\optimization_results_{tuning_method}_without_dropout.csv")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(fr"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\LSTMCV\optimization_results_{tuning_method}_without_dropout_original.csv", index=False)

    if len(results_list) > 0:
        best_config = min(results_list, key=lambda x: x['test_rmse'])
        print("\nBest Configuration:")
        for key, value in best_config.items():
            print(f"{key}: {value}")
