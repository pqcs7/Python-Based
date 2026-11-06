from pathlib import Path

import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas.api.types as ptypes

from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler, PowerTransformer, QuantileTransformer
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#--------------------------------------------
def load_data(file_path, date_col="Date"):
    df = pd.read_csv(file_path, parse_dates=[date_col])
    # Replace placeholder missing values and drop fully empty rows
    df = df.replace('---', np.nan)
    df = df.dropna().reset_index(drop=True)
    return df

#--------------------------------------------
def add_wind_sincos(df, wind_col="DWD"):
    wind_dir_map = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
    }
    if wind_col in df.columns:
        df = df.copy()
        df["DWD_deg"] = df[wind_col].map(wind_dir_map)
        theta = np.deg2rad(df["DWD_deg"])
        df["DWD_sin"] = np.sin(theta)
        df["DWD_cos"] = np.cos(theta)
        df = df.drop(columns=[wind_col, "DWD_deg"])
    return df

#--------------------------------------------
def split_by_date_masks(df, date_col="Date"):
    dates = pd.to_datetime(df[date_col])
    train_mask = (dates >= "2010-01-01") & (dates <= "2019-12-31")
    val_mask  = (dates >= "2020-01-01") & (dates <= "2021-12-31")
    test_mask   = (dates >= "2022-01-01")
    return train_mask, test_mask, val_mask

#--------------------------------------------
def get_default_scalers():
    return {
        "temp": PowerTransformer(method="yeo-johnson"),
        "rh": QuantileTransformer(output_distribution="uniform", random_state=42),
        "prs": QuantileTransformer(output_distribution="normal", random_state=42),
        "rain": ("log1p", MinMaxScaler()),
        "wind": RobustScaler()
    }

def fit_and_transform_scalers(df, train_mask, test_mask, val_mask, config,
                              temp_cols, rh_cols, prs_cols, rain_col, wind_cols, date_col="Date"):
    train = df.loc[train_mask].copy()
    test  = df.loc[test_mask].copy()
    val   = df.loc[val_mask].copy()

    has_date = date_col in train.columns
    if has_date:
        train_dates = train[date_col].copy()
        test_dates = test[date_col].copy()
        val_dates = val[date_col].copy()
        train = train.drop(columns=[date_col])
        test = test.drop(columns=[date_col])
        val = val.drop(columns=[date_col])

    train = train.apply(pd.to_numeric, errors='coerce')
    test  = test.apply(pd.to_numeric, errors='coerce')
    val   = val.apply(pd.to_numeric, errors='coerce')

    train = train.dropna().reset_index(drop=True)
    test  = test.dropna().reset_index(drop=True)
    val   = val.dropna().reset_index(drop=True)

    temp_scaler = config["temp"]
    temp_scaler.fit(train[temp_cols])
    train[temp_cols] = temp_scaler.transform(train[temp_cols])
    test[temp_cols]  = temp_scaler.transform(test[temp_cols])
    val[temp_cols]   = temp_scaler.transform(val[temp_cols])

    rh_scaler = config["rh"]
    rh_scaler.fit(train[rh_cols])
    train[rh_cols] = rh_scaler.transform(train[rh_cols])
    test[rh_cols]  = rh_scaler.transform(test[rh_cols])
    val[rh_cols]   = rh_scaler.transform(val[rh_cols])

    prs_scaler = config["prs"]
    prs_scaler.fit(train[prs_cols])
    train[prs_cols] = prs_scaler.transform(train[prs_cols])
    test[prs_cols]  = prs_scaler.transform(test[prs_cols])
    val[prs_cols]   = prs_scaler.transform(val[prs_cols])

    mode, rain_scaler = config["rain"]
    if mode == "log1p":
        train_rain = np.log1p(train[rain_col])
        rain_scaler.fit(train_rain)
        train[rain_col] = rain_scaler.transform(train_rain)
        test[rain_col]  = rain_scaler.transform(np.log1p(test[rain_col]))
        val[rain_col]   = rain_scaler.transform(np.log1p(val[rain_col]))
    else:
        rain_scaler.fit(train[rain_col])
        train[rain_col] = rain_scaler.transform(train[rain_col])
        test[rain_col]  = rain_scaler.transform(test[rain_col])
        val[rain_col]   = rain_scaler.transform(val[rain_col])

    wind_scaler = config["wind"]
    wind_scaler.fit(train[wind_cols])
    train[wind_cols] = wind_scaler.transform(train[wind_cols])
    test[wind_cols]  = wind_scaler.transform(test[wind_cols])
    val[wind_cols]   = wind_scaler.transform(val[wind_cols])

    if has_date:
        train[date_col] = train_dates.iloc[:len(train)].values
        test[date_col]  = test_dates.iloc[:len(test)].values
        val[date_col]   = val_dates.iloc[:len(val)].values

    fitted_scalers = {
        "temp": temp_scaler,
        "rh": rh_scaler,
        "prs": prs_scaler,
        "rain": rain_scaler,
        "wind": wind_scaler,
        "rain_mode": mode
    }
    return train, test, val, fitted_scalers

#--------------------------------------------
def create_dataset_multi_from_df(df_subset, look_back, date_col="Date"):
    if date_col in df_subset.columns:
        df_num = df_subset.drop(columns=[date_col]).reset_index(drop=True)
        target_dates = df_subset[date_col].reset_index(drop=True).iloc[look_back:].reset_index(drop=True)
    else:
        df_num = df_subset.reset_index(drop=True)
        target_dates = None
    arr = df_num.values.astype(np.float32)
    X, y = [], []
    for i in range(len(arr) - look_back):
        X.append(arr[i:i + look_back])
        y.append(arr[i + look_back])
    return np.array(X), np.array(y), df_num.columns.tolist(), target_dates

#--------------------------------------------
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.BatchNormalization(),
        layers.LSTM(32, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dense(16, activation='relu'),
        layers.Dense(input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#--------------------------------------------
def train_model(X_train, y_train, X_val, y_val, epochs, batch_size):
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history

#--------------------------------------------
def inverse_transform_predictions(y_scaled, df_columns, fitted_scalers,
                                  temp_cols, rh_cols, prs_cols, rain_col, wind_cols):
    df_pred_scaled = pd.DataFrame(y_scaled, columns=df_columns)
    df_pred = df_pred_scaled.copy()
    df_pred[temp_cols] = fitted_scalers["temp"].inverse_transform(df_pred[temp_cols])
    df_pred[rh_cols] = fitted_scalers["rh"].inverse_transform(df_pred[rh_cols])
    df_pred[prs_cols] = fitted_scalers["prs"].inverse_transform(df_pred[prs_cols])
    if fitted_scalers["rain_mode"] == "log1p":
        df_pred[rain_col] = np.expm1(fitted_scalers["rain"].inverse_transform(df_pred[rain_col]))
    else:
        df_pred[rain_col] = fitted_scalers["rain"].inverse_transform(df_pred[rain_col])
    df_pred[wind_cols] = fitted_scalers["wind"].inverse_transform(df_pred[wind_cols])
    return df_pred

#--------------------------------------------
def autoregressive_forecast(model, last_window_scaled, steps, fitted_scalers, df_columns,
                            temp_cols, rh_cols, prs_cols, rain_col, wind_cols):
    predictions_scaled = []
    current_window = last_window_scaled.copy()
    for _ in range(steps):
        next_scaled = model.predict(current_window[np.newaxis, :, :], verbose=0)[0]
        predictions_scaled.append(next_scaled)
        current_window = np.vstack([current_window[1:], next_scaled])
    df_future_scaled = pd.DataFrame(predictions_scaled, columns=df_columns)
    df_future = inverse_transform_predictions(df_future_scaled.values, df_columns, fitted_scalers,
                                              temp_cols, rh_cols, prs_cols, rain_col, wind_cols)
    return df_future


def plot_true_vs_pred(df_true, df_pred, dates=None, title_prefix="Validation", figsize=(12, 3)):

    cols = df_true.columns.tolist()
    n = len(cols)
    plt.figure(figsize=(14, 3 * n))
    for i, col in enumerate(cols):
        plt.subplot(n, 1, i+1)
        if dates is not None:
            plt.plot(dates, df_true[col].values, label=f"{col} (True)", color='blue')
            plt.plot(dates, df_pred[col].values, label=f"{col} (Pred)", color='red', alpha=0.8)
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
            plt.gcf().autofmt_xdate()
        else:
            plt.plot(df_true[col].values, label=f"{col} (True)", color='blue')
            plt.plot(df_pred[col].values, label=f"{col} (Pred)", color='red', alpha=0.8)
        plt.title(f"{title_prefix} — {col}")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

#--------------------------------------------
def main(file_path):
    temp_cols = ["T_mean", "T_max", "T_min"]
    rh_cols   = ["RH_mean", "RH_max", "RH_min"]
    prs_cols  = ["Prs_mean", "Prs_max", "Prs_min"]
    rain_col  = ["Ac_R"]
    wind_cols = ["WS_mean", "WG"]

    df = load_data(file_path, date_col="Date")
    df = add_wind_sincos(df, wind_col="DWD")

    train_mask, test_mask, val_mask = split_by_date_masks(df, date_col="Date")
    print(f"Total rows: {len(df)}, Train: {train_mask.sum()}, Test: {test_mask.sum()}, Val: {val_mask.sum()}")

    config = get_default_scalers()
    train_scaled, test_scaled, val_scaled, fitted_scalers = fit_and_transform_scalers(
        df, train_mask, test_mask, val_mask, config,
        temp_cols=temp_cols, rh_cols=rh_cols, prs_cols=prs_cols,
        rain_col=rain_col, wind_cols=wind_cols, date_col="Date"
    )

    look_back = 21
    X_train, y_train, columns, _       = create_dataset_multi_from_df(train_scaled, look_back=look_back, date_col="Date")
    X_test,  y_test, _, test_target_dates = create_dataset_multi_from_df(test_scaled,  look_back=look_back, date_col="Date")
    X_val,   y_val, _, val_target_dates  = create_dataset_multi_from_df(val_scaled,   look_back=look_back, date_col="Date")

    if len(X_train) == 0:
        raise ValueError("Training set too small for the chosen look_back.")
    if len(X_test) == 0:
        raise ValueError("Test set too small for the chosen look_back.")
    if len(X_val) == 0:
        print("Warning: Validation set too small for windowing; validation will be skipped in training.")


    model, history = train_model(X_train, y_train, X_test, y_test, epochs=40, batch_size=32)

    # Plot training/validation loss
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("LSTM Training Loss (Adam)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


    y_pred_test = model.predict(X_test, verbose=0)
    mae_per_param_test = np.mean(np.abs(y_pred_test - y_test), axis=0)
    mse_per_param_test = np.mean((y_pred_test - y_test)**2, axis=0)

    results = {
        "columns": columns,
        "mae_per_param_test": mae_per_param_test,
        "mse_per_param_test": mse_per_param_test,
        "mae_total_test": np.mean(mae_per_param_test),
        "mse_total_test": np.mean(mse_per_param_test),
        "history": history
    }

    print("\nTest MAE per variable:")
    for col, val in zip(columns, mae_per_param_test):
        print(f"  {col}: {val:.4f}")
    print(f"Total MAE (test): {results['mae_total_test']:.4f}")

    # If validation windows exist: inverse-transform and plot true vs predicted
    if len(X_val) > 0:
        y_pred_val = model.predict(X_val, verbose=0)
        df_pred_val = inverse_transform_predictions(y_pred_val, columns, fitted_scalers,
                                                    temp_cols, rh_cols, prs_cols, rain_col, wind_cols)
        df_true_val = inverse_transform_predictions(y_val, columns, fitted_scalers,
                                                    temp_cols, rh_cols, prs_cols, rain_col, wind_cols)

        plot_true_vs_pred(df_true_val, df_pred_val, dates=val_target_dates, title_prefix="Validation")

        results.update({
            "df_pred_val": df_pred_val,
            "df_true_val": df_true_val
        })
    else:
        print("No validation windows to plot.")


    last_window = X_val[-1] if len(X_val) > 0 else X_test[-1]
    forecast_hours = 24
    df_future = autoregressive_forecast(model, last_window, forecast_hours, fitted_scalers,
                                        columns, temp_cols, rh_cols, prs_cols, rain_col, wind_cols)

    forecast_start = None
    if val_target_dates is not None and len(val_target_dates) > 0:
        last_target_date = val_target_dates.iloc[-1]
        forecast_start = last_target_date + pd.Timedelta(hours=1)
    elif test_target_dates is not None and len(test_target_dates) > 0:
        last_target_date = test_target_dates.iloc[-1]
        forecast_start = last_target_date + pd.Timedelta(hours=1)


    results["df_future"] = df_future

    return results

#--------------------------------------------
if __name__ == "__main__":
    file_path = Path(__file__).with_name("athens.csv")
    downloaded_path = gdown.download(
        id="1jGSfgPkq_D144PdLDtfLVz7W6VXONWwu",
        output=str(file_path),
        quiet=False,
    )
    if downloaded_path is None:
        raise RuntimeError(
            "Could not download athens.csv. Check that the Google Drive file "
            "is shared with anyone who has the link."
        )
    results = main(downloaded_path)