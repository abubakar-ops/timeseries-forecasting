import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from time import time
import os

from src.nf import MODELS, forecast_pretrained_model
from src.model_descriptions import model_cards
from datasetsforecast.losses import mae, mape, rmse, smape

DATASETS = {
    "Electricity (Ercot COAST)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/ercot_COAST.csv",
    "Web Traffic (Peyton Manning)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/peyton_manning.csv",
    "Demand (AirPassengers)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv",
    "Finance (Exchange USD-EUR)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/usdeur.csv",
}

def plot(df, uid, df_forecast, model):
    figs = []
    figs.append(
        go.Scatter(
            x=df["ds"],
            y=df["y"],
            mode="lines",
            marker=dict(color="#236796"),
            legendrank=1,
            name=uid,
        )
    )
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
                legendrank=5,
                opacity=0.5,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=ds_f,
                y=df_forecast["forecast"],
                mode="lines",
                legendrank=4,
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

def run_forecast(data_selection, url_input, uploaded_file, timestamp_col, value_col, model_name, fh, max_steps, cv_windows):

    # Determine the data source
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file.name)
        # Ensure timestamp_col and value_col are not None before renaming
        if timestamp_col and value_col:
            df = df.rename(columns={timestamp_col: "ds", value_col: "y"})
        else:
            # Handle case where columns are not selected
            # You might want to return an error message here
            return go.Figure(), pd.DataFrame(), pd.DataFrame(), "Error: Please select timestamp and value columns for the uploaded file.", None, go.Figure(), pd.DataFrame()

    elif url_input:
        df = pd.read_csv(url_input)
        # For URL, assume the first two columns are the correct ones if not named ds/y
        if "ds" not in df.columns or "y" not in df.columns:
            df = df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})
    
    else:
        df = pd.read_csv(DATASETS[data_selection])
        if "timestamp" in df.columns and "value" in df.columns:
            df = df.rename(columns={"timestamp": "ds", "value": "y"})
        elif "ds" not in df.columns or "y" not in df.columns:
            df = df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})


    if "unique_id" not in df.columns:
        df.insert(0, "unique_id", "ts_0")

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    uid = df["unique_id"].unique()[0]
    model_file = MODELS[model_name]["model"]

    # ---- Forecast Tab Logic ----
    init = time()
    df_forecast = forecast_pretrained_model(df, model_file, fh, max_steps)
    end = time()
    df_forecast = df_forecast.rename(columns={"y_5": "forecast_lo_90", "y_50": "forecast", "y_95": "forecast_hi_90"})

    forecast_plot = plot(df.query("unique_id == @uid"), uid, df_forecast.query("unique_id == @uid"), model_name)
    inference_time_message = f'Done! Approximate inference time CPU: {0.7*(end-init):.2f} seconds.'

    # Save forecast to a temporary file for download
    forecast_filepath = "forecasts.csv"
    df_forecast.to_csv(forecast_filepath, index=False)

    # ---- Cross Validation Tab Logic ----
    df_cv_forecast_list = []
    for i_window in range(cv_windows, 0, -1):
        test = df.groupby("unique_id").tail(i_window * fh)
        train = df.drop(test.index)
        if not train.empty:
            df_forecast_w = forecast_pretrained_model(train, model_file, fh, max_steps)
            df_forecast_w = df_forecast_w.rename(columns={"y_5": "forecast_lo_90", "y_50": "forecast", "y_95": "forecast_hi_90"})
            df_forecast_w.insert(2, "window", i_window)
            df_cv_forecast_list.append(df_forecast_w)

    if df_cv_forecast_list:
        df_cv_forecast = pd.concat(df_cv_forecast_list)
        df_cv_forecast["ds"] = pd.to_datetime(df_cv_forecast["ds"])
        df_cv_forecast = df_cv_forecast.merge(df, how="left", on=["unique_id", "ds"])

        metrics = [mae, mape, rmse, smape]
        evaluation = df_cv_forecast.groupby(["unique_id", "window"]).apply(
            lambda x: [f'{fn(x["y"].values, x["forecast"].values):.2f}' for fn in metrics] if x['y'].notna().all() else ['N/A'] * len(metrics)
        )
        evaluation = evaluation.rename("eval").reset_index()
        evaluation["eval"] = evaluation["eval"].str.join(",")
        evaluation[["MAE", "MAPE", "RMSE", "sMAPE"]] = evaluation["eval"].str.split(",", expand=True)

        cv_plot = plot(df.query("unique_id == @uid"), uid, df_cv_forecast.query("unique_id == @uid"), model_name)
        cv_metrics_df = evaluation.query("unique_id == @uid").drop(columns=["unique_id", "eval"]).set_index("window")
    else:
        cv_plot = go.Figure() # Empty plot
        cv_metrics_df = pd.DataFrame() # Empty dataframe

    return forecast_plot, df, df_forecast, inference_time_message, forecast_filepath, cv_plot, cv_metrics_df

# --- New function to update dropdowns ---
def update_column_dropdowns(file):
    if file is not None:
        try:
            df = pd.read_csv(file.name, nrows=1) # Read only header to be fast
            columns = df.columns.tolist()
            # Intelligently guess default values
            ts_guess = next((col for col in ['ds', 'timestamp', 'date'] if col in columns), columns[0] if columns else None)
            val_guess = next((col for col in ['y', 'value', 'values'] if col in columns), columns[1] if len(columns) > 1 else None)
            return gr.Dropdown(choices=columns, value=ts_guess, interactive=True), gr.Dropdown(choices=columns, value=val_guess, interactive=True)
        except Exception as e:
            print(f"Error reading CSV to get columns: {e}")
            return gr.Dropdown(choices=[], interactive=False), gr.Dropdown(choices=[], interactive=False)
    return gr.Dropdown(choices=[], interactive=False), gr.Dropdown(choices=[], interactive=False)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Transfer Learning: Revolutionizing Time Series by Nixtla")
    gr.Markdown(
        """
        The success of startups like Open AI and Stability highlights the potential for transfer learning (TL) techniques to have a similar impact on the field of time series forecasting.
        TL can achieve lightning-fast predictions with a fraction of the computational cost by pre-training a flexible model on a large dataset and then using it on another dataset with little to no additional training.
        In this live demo, you can use pre-trained models by Nixtla (trained on the M4 dataset) to predict your own datasets. You can also see how the models perform on unseen example datasets.
        """
    )

    with gr.Accordion("Step 1: Configure Your Dataset and Model", open=True):
        with gr.Row():
            with gr.Column():
                data_selection = gr.Dropdown(label="Select example dataset", choices=list(DATASETS.keys()), value="Electricity (Ercot COAST)")
                url_input = gr.Textbox(label="Or provide your own URL to a CSV file")
                uploaded_file = gr.File(label="Or upload a CSV file")
                gr.Markdown("For uploaded files, specify the column names below.")
                # --- Changed from Textbox to Dropdown ---
                timestamp_col_dd = gr.Dropdown(label="Timestamp column", interactive=False)
                value_col_dd = gr.Dropdown(label="Value column", interactive=False)

            with gr.Column():
                model_name = gr.Dropdown(label="Select your model", choices=list(MODELS.keys()), value="Pretrained N-HiTS M4 Hourly")
                fh = gr.Slider(label="Forecast horizon", minimum=1, maximum=100, value=18, step=1)
                max_steps = gr.Slider(label="N-shot inference (fine-tuning steps)", minimum=0, maximum=100, value=0, step=1)
                submit_btn = gr.Button("Submit")

    # --- New event handler for file upload ---
    uploaded_file.upload(
        fn=update_column_dropdowns,
        inputs=[uploaded_file],
        outputs=[timestamp_col_dd, value_col_dd]
    )

    with gr.Tabs():
        with gr.TabItem("📈 Forecast"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Data")
                    input_df_display = gr.Dataframe(interactive=False)
                    gr.Markdown("### Forecast Data")
                    forecast_df_display = gr.Dataframe(interactive=False)
                    forecast_download_btn = gr.DownloadButton(label="Download Forecasts as CSV", visible=True)
                with gr.Column(scale=2):
                    inference_time_display = gr.Markdown()
                    forecast_plot_display = gr.Plot()

        with gr.TabItem("🔎 Cross Validation"):
            with gr.Row():
                with gr.Column(scale=1):
                    cv_windows = gr.Slider(label="Cross validation windows", minimum=1, maximum=10, value=1, step=1)
                    gr.Markdown("### Evaluation Metrics")
                    cv_metrics_display = gr.Dataframe(interactive=False)
                with gr.Column(scale=2):
                    cv_plot_display = gr.Plot()

        with gr.TabItem("📚 Documentation"):
            with gr.Tabs():
                with gr.TabItem("🚀 Transfer Learning"):
                    gr.Markdown("""
                        ...
                        """)
                with gr.TabItem("🔎 Description of the model"):
                    model_card_selection = gr.Dropdown(label="Select a model to see its description", choices=list(model_cards.keys()), value="nhitsh")
                    model_card_display = gr.Markdown()

                    def update_model_card(selection):
                        card = model_cards.get(selection, {})
                        return f"""
                        ### Abstract
                        {card.get('Abstract', 'N/A')}
                        ### Intended use
                        {card.get('Intended use', 'N/A')}
                        ### Limitations
                        {card.get('Limitations', 'N/A')}
                        ### Training data
                        {card.get('Training data', 'N/A')}
                        ### Citation Info
                        ```
                        {card.get('Citation Info', 'N/A')}
                        ```
                        """
                    model_card_selection.change(fn=update_model_card, inputs=model_card_selection, outputs=model_card_display)
                    demo.load(fn=update_model_card, inputs=model_card_selection, outputs=model_card_display)

                with gr.TabItem("📚 References"):
                    gr.Markdown("""
                        ...
                        """)

        with gr.TabItem("🔮 Nixtlaverse"):
            gr.Markdown("""
                ...
                """)
            gr.Image("https://files.readme.io/168cdb2-Screen_Shot_2022-09-30_at_10.40.09.png", width=800)

    submit_btn.click(
        fn=run_forecast,
        inputs=[data_selection, url_input, uploaded_file, timestamp_col_dd, value_col_dd, model_name, fh, max_steps, cv_windows],
        outputs=[forecast_plot_display, input_df_display, forecast_df_display, inference_time_display, forecast_download_btn, cv_plot_display, cv_metrics_display]
    )

if __name__ == "__main__":
    demo.launch(share=True)
