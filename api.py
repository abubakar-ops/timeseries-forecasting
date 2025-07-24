
import pandas as pd
import plotly.graph_objects as go
from time import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, Form, UploadFile
from datasetsforecast.losses import mae, mape, rmse, smape

# This import is preserved as requested.
from src.nf import MODELS, forecast_pretrained_model

# --- Constants and Configuration ---
DATASETS = {
    "Electricity (Ercot COAST)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/ercot_COAST.csv",
    "Web Traffic (Peyton Manning)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/peyton_manning.csv",
    "Demand (AirPassengers)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv",
    "Finance (Exchange USD-EUR)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/usdeur.csv",
}

# --- Core Logic (Unchanged) ---

def plot(df, uid, df_forecast, model):
    """Creates a Plotly figure for the forecast."""
    figs = [
        go.Scatter(
            x=df["ds"],
            y=df["y"],
            mode="lines",
            marker=dict(color="#236796"),
            name=uid,
        )
    ]
    if df_forecast is not None and not df_forecast.empty:
        ds_f = df_forecast["ds"].to_list()
        lo = df_forecast["forecast_lo_90"].to_list()
        hi = df_forecast["forecast_hi_90"].to_list()
        figs.extend([
            go.Scatter(
                x=ds_f + ds_f[::-1],
                y=hi + lo[::-1],
                fill="toself",
                fillcolor="#E7C4C0",
                mode="lines",
                line=dict(color="#E7C4C0"),
                name="Prediction Intervals (90%)",
                opacity=0.5,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=ds_f,
                y=df_forecast["forecast"],
                mode="lines",
                marker=dict(color="#E7C4C0"),
                name=f"Forecast {uid}",
            ),
        ])
    fig = go.Figure(figs)
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        title=f"Forecasts for {uid} using Transfer Learning (from {model})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, b=20),
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    if not df.empty and df_forecast is not None and not df_forecast.empty:
        initial_range = [df.tail(200)["ds"].iloc[0], ds_f[-1]]
        fig["layout"]["xaxis"].update(range=initial_range)
    return fig

def process_forecast(df: pd.DataFrame, model_name: str, fh: int, max_steps: int, cv_windows: int):
    # (This function is unchanged)
    if "unique_id" not in df.columns:
        df.insert(0, "unique_id", "ts_0")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])
    uid = df["unique_id"].unique()[0]
    model_file = MODELS[model_name]["model"]
    init = time()
    df_forecast = forecast_pretrained_model(df, model_file, fh, max_steps)
    end = time()
    df_forecast = df_forecast.rename(columns={"y_5": "forecast_lo_90", "y_50": "forecast", "y_95": "forecast_hi_90"})
    forecast_plot = plot(df.query("unique_id == @uid"), uid, df_forecast.query("unique_id == @uid"), model_name)
    inference_time_message = f'Approximate inference time CPU: {0.7*(end-init):.2f} seconds.'
    df_cv_forecast_list = []
    if cv_windows > 0:
      for i_window in range(cv_windows, 0, -1):
          test = df.groupby("unique_id").tail(i_window * fh)
          train = df.drop(test.index)
          if not train.empty:
              df_fcst = forecast_pretrained_model(train, model_file, fh, max_steps)
              df_fcst = df_fcst.rename(columns={"y_5": "forecast_lo_90", "y_50": "forecast", "y_95": "forecast_hi_90"})
              df_fcst.insert(2, "window", i_window)
              df_cv_forecast_list.append(df_fcst)
    if df_cv_forecast_list:
        df_cv_forecast = pd.concat(df_cv_forecast_list)
        df_cv_forecast["ds"] = pd.to_datetime(df_cv_forecast["ds"])
        df_cv_forecast = df_cv_forecast.merge(df, how="left", on=["unique_id", "ds"])
        evaluation = df_cv_forecast.groupby(["unique_id", "window"]).apply(lambda x: pd.Series([round(mae(x['y'].values, x['forecast'].values), 2), round(mape(x['y'].values, x['forecast'].values), 2), round(rmse(x['y'].values, x['forecast'].values), 2), round(smape(x['y'].values, x['forecast'].values), 2)], index=["MAE", "MAPE", "RMSE", "sMAPE"])).reset_index()
        cv_plot = plot(df.query("unique_id == @uid"), uid, df_cv_forecast.query("unique_id == @uid"), model_name)
        cv_metrics_df = evaluation.query("unique_id == @uid").set_index("window")
    else:
        cv_plot = go.Figure()
        cv_metrics_df = pd.DataFrame()
    return forecast_plot, df_forecast, inference_time_message, cv_plot, cv_metrics_df

# --- FastAPI Application ---

app = FastAPI(title="Time Series Forecasting API")

# --- Unified API Endpoint ---

@app.post("/forecast/", response_model=Dict[str, Any])
async def create_forecast(
    # File, URL, and Dataset Selection are all optional
    file: Optional[UploadFile] = File(None),
    url_input: Optional[str] = Form(None),
    data_selection: Optional[str] = Form("Electricity (Ercot COAST)"),
    # Other parameters are sent as form fields
    model_name: str = Form("Pretrained N-HiTS M4 Hourly"),
    fh: int = Form(18),
    max_steps: int = Form(0),
    cv_windows: int = Form(1),
    timestamp_col: Optional[str] = Form(None),
    value_col: Optional[str] = Form(None)
):
    """
    Unified endpoint to generate time series forecasts.
    Accepts data from a file upload, a URL, or a predefined dataset.
    All parameters must be sent as multipart/form-data.
    """
    # Determine the data source with a clear priority: file > url > selection
    if file:
        df = pd.read_csv(file.file)
    elif url_input:
        df = pd.read_csv(url_input)
    else:
        df = pd.read_csv(DATASETS[data_selection])

    # Standardize column names after loading the dataframe
    if "ds" not in df.columns or "y" not in df.columns:
        if timestamp_col and value_col:
             df = df.rename(columns={timestamp_col: "ds", value_col: "y"})
        elif "timestamp" in df.columns and "value" in df.columns:
             df = df.rename(columns={"timestamp": "ds", "value": "y"})
        else:
             df = df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})

    # Process the forecast using the prepared dataframe
    forecast_plot, df_forecast, inference_time, cv_plot, eval_df = process_forecast(
        df, model_name, fh, max_steps, cv_windows
    )

    # Return the results
    return {
        "inference_time": inference_time,
        "forecast_data": df_forecast.to_dict(orient='records'),
        "evaluation_metrics": eval_df.reset_index().to_dict(orient='records'),
        "forecast_plot_json": forecast_plot.to_json(),
        "cv_plot_json": cv_plot.to_json(),
    }
