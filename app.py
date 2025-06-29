from time import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datasetsforecast.losses import rmse, mae, smape, mse, mape
from st_aggrid import AgGrid

from src.nf import MODELS, forecast_pretrained_model
from src.model_descriptions import model_cards

DATASETS = {
    "Electricity (Ercot COAST)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/ercot_COAST.csv",
    #"Electriciy (ERCOT, multiple markets)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/ercot_multiple_ts.csv",
    "Web Traffic (Peyton Manning)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/peyton_manning.csv",
    "Demand (AirPassengers)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv",
    "Finance (Exchange USD-EUR)": "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/usdeur.csv",
}


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


def plot(df, uid, df_forecast, model):
    figs = []
    figs += [
        go.Scatter(
            x=df["ds"],
            y=df["y"],
            mode="lines",
            marker=dict(color="#236796"),
            legendrank=1,
            name=uid,
        ),
    ]
    if df_forecast is not None:
        ds_f = df_forecast["ds"].to_list()
        lo = df_forecast["forecast_lo_90"].to_list()
        hi = df_forecast["forecast_hi_90"].to_list()
        figs += [
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
        ]
    fig = go.Figure(figs)
    fig.update_layout(
        {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"}
    )
    fig.update_layout(
        title=f"Forecasts for {uid} using Transfer Learning (from {model})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, b=20),
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    initial_range = [df.tail(200)["ds"].iloc[0], ds_f[-1]]
    fig["layout"]["xaxis"].update(range=initial_range)
    return fig


def st_transfer_learning():
    st.set_page_config(
        page_title="Time Series Visualization",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(
        "Transfer Learning: Revolutionizing Time Series by Nixtla"
    )
    st.write(
        "<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True
    )

    intro = """
        The success of startups like Open AI and Stability highlights the potential for transfer learning (TL) techniques to have a similar impact on the field of time series forecasting.

        TL can achieve lightning-fast predictions with a fraction of the computational cost by pre-training a flexible model on a large dataset and then using it on another dataset with little to no additional training.

        In this live demo, you can use pre-trained models by Nixtla (trained on the M4 dataset) to predict your own datasets. You can also see how the models perform on unseen example datasets.
	"""
    st.write(intro)

    required_cols = ["ds", "y"]

    with st.sidebar.expander("Dataset", expanded=False):
        data_selection = st.selectbox("Select example dataset", DATASETS.keys())
        data_url = DATASETS[data_selection]
        url_json = st.text_input("Data (you can pass your own url here)", data_url)
        st.write(
            "You can also upload a CSV file like [this one](https://github.com/Nixtla/transfer-learning-time-series/blob/main/datasets/air_passengers.csv)."
        )

        uploaded_file = st.file_uploader("Upload CSV")
        with st.form("Data"):

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                cols = df.columns
                timestamp_col = st.selectbox("Timestamp column", options=cols)
                value_col = st.selectbox("Value column", options=cols)
            else:
                timestamp_col = st.text_input("Timestamp column", value="timestamp")
                value_col = st.text_input("Value column", value="value")
            st.write("You must press Submit each time you want to forecast.")
            submitted = st.form_submit_button("Submit")
            if submitted:
                if uploaded_file is None:
                    st.write("Please provide a dataframe.")
                    if url_json.endswith("json"):
                        df = pd.read_json(url_json)
                    else:
                        df = pd.read_csv(url_json)
                    df = df.rename(
                        columns=dict(zip([timestamp_col, value_col], required_cols))
                    )
                else:
                    # df = pd.read_csv(uploaded_file)
                    df = df.rename(
                        columns=dict(zip([timestamp_col, value_col], required_cols))
                    )
            else:
                if url_json.endswith("json"):
                    df = pd.read_json(url_json)
                else:
                    df = pd.read_csv(url_json)
                cols = df.columns
                if "unique_id" in cols:
                    cols = cols[-2:]
                df = df.rename(columns=dict(zip(cols, required_cols)))

            if "unique_id" not in df:
                df.insert(0, "unique_id", "ts_0")

            df["ds"] = pd.to_datetime(df["ds"])
            df = df.sort_values(["unique_id", "ds"])

    with st.sidebar:
        st.write("Define the pretrained model you want to use to forecast your data")
        model_name = st.selectbox("Select your model", tuple(MODELS.keys()))
        model_file = MODELS[model_name]["model"]
        st.write("Choose how many steps you want to forecast")
        fh = st.number_input("Forecast horizon", value=18)
        st.write(
            "Choose for how many steps the pretrained model will be updated using your data (use 0 for fast computation)"
        )
        max_steps = st.number_input("N-shot inference", value=0)

    # tabs
    tab_fcst, tab_cv, tab_docs, tab_nixtla = st.tabs(
        [
            "📈 Forecast",
            "🔎 Cross Validation",
            "📚 Documentation",
            "🔮 Nixtlaverse",
        ]
    )

    uids = df["unique_id"].unique()
    fcst_cols = ["forecast_lo_90", "forecast", "forecast_hi_90"]

    with tab_fcst:
        uid = uids[0]#st.selectbox("Dataset", options=uids)
        col1, col2 = st.columns([2, 4])
        with col1:
            tab_insample, tab_forecast = st.tabs(
                ["Modify input data", "Modify forecasts"]
            )
            with tab_insample:
                df_grid = df.query("unique_id == @uid").drop(columns="unique_id")
                grid_table = AgGrid(
                    df_grid,
                    editable=True,
                    theme="streamlit",
                    fit_columns_on_grid_load=True,
                    height=360,
                )
                df.loc[df["unique_id"] == uid, "y"] = (
                    grid_table["data"].sort_values("ds")["y"].values
                )
                # forecast code
                init = time()
                df_forecast = forecast_pretrained_model(df, model_file, fh, max_steps)
                end = time()
                df_forecast = df_forecast.rename(
                    columns=dict(zip(["y_5", "y_50", "y_95"], fcst_cols))
                )
            with tab_forecast:
                df_fcst_grid = df_forecast.query("unique_id == @uid").filter(
                    ["ds", "forecast"]
                )
                grid_fcst_table = AgGrid(
                    df_fcst_grid,
                    editable=True,
                    theme="streamlit",
                    fit_columns_on_grid_load=True,
                    height=360,
                )
                changes = (
                    df_forecast.query("unique_id == @uid")["forecast"].values
                    - grid_fcst_table["data"].sort_values("ds")["forecast"].values
                )
                for col in fcst_cols:
                    df_forecast.loc[df_forecast["unique_id"] == uid, col] = (
                        df_forecast.loc[df_forecast["unique_id"] == uid, col] - changes
                    )
        with col2:
            st.plotly_chart(
                plot(
                    df.query("unique_id == @uid"),
                    uid,
                    df_forecast.query("unique_id == @uid"),
                    model_name,
                ),
                use_container_width=True,
            )
            st.success(f'Done! Approximate inference time CPU: {0.7*(end-init):.2f} seconds.')

    with tab_cv:
        col_uid, col_n_windows = st.columns(2)
        uid = uids[0]
        #with col_uid:
        #    uid = st.selectbox("Time series to analyse", options=uids, key="uid_cv")
        with col_n_windows:
            n_windows = st.number_input("Cross validation windows", value=1)
        df_forecast = []
        for i_window in range(n_windows, 0, -1):
            test = df.groupby("unique_id").tail(i_window * fh)
            df_forecast_w = forecast_pretrained_model(
                df.drop(test.index), model_file, fh, max_steps
            )
            df_forecast_w = df_forecast_w.rename(
                columns=dict(zip(["y_5", "y_50", "y_95"], fcst_cols))
            )
            df_forecast_w.insert(2, "window", i_window)
            df_forecast.append(df_forecast_w)
        df_forecast = pd.concat(df_forecast)
        df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])
        df_forecast = df_forecast.merge(df, how="left", on=["unique_id", "ds"])
        metrics = [mae, mape, rmse, smape]
        evaluation = df_forecast.groupby(["unique_id", "window"]).apply(
            lambda df: [f'{fn(df["y"].values, df["forecast"]):.2f}' for fn in metrics]
        )
        evaluation = evaluation.rename("eval").reset_index()
        evaluation["eval"] = evaluation["eval"].str.join(",")
        evaluation[["MAE", "MAPE", "RMSE", "sMAPE"]] = evaluation["eval"].str.split(
            ",", expand=True
        )
        col_eval, col_plot = st.columns([2, 4])
        with col_eval:
            st.write("Evaluation metrics for each cross validation window")
            st.dataframe(
                evaluation.query("unique_id == @uid")
                .drop(columns=["unique_id", "eval"])
                .set_index("window")
            )
        with col_plot:
            st.plotly_chart(
                plot(
                    df.query("unique_id == @uid"),
                    uid,
                    df_forecast.query("unique_id == @uid").drop(columns="y"),
                    model_name,
                ),
                use_container_width=True,
            )
    with tab_docs:
        tab_transfer, tab_desc, tab_ref = st.tabs(
            [
                "🚀 Transfer Learning",
                "🔎 Description of the model",
                "📚 References",
            ]
        )

        with tab_desc:
            model_card_name = MODELS[model_name]["card"]
            st.subheader("Abstract")
            st.write(f"""{model_cards[model_card_name]['Abstract']}""")
            st.subheader("Intended use")
            st.write(f"""{model_cards[model_card_name]['Intended use']}""")
            st.subheader("Secondary use")
            st.write(f"""{model_cards[model_card_name]['Secondary use']}""")
            st.subheader("Limitations")
            st.write(f"""{model_cards[model_card_name]['Limitations']}""")
            st.subheader("Training data")
            st.write(f"""{model_cards[model_card_name]['Training data']}""")
            st.subheader("BibTex/Citation Info")
            st.code(f"""{model_cards[model_card_name]['Citation Info']}""")

        with tab_transfer:
            transfer_text = """
                Transfer learning refers to the process of pre-training a flexible model on a large dataset and using it later on other data with little to no training. It is one of the most outstanding 🚀 achievements in Machine Learning 🧠 and has many practical applications.

                For time series forecasting, the technique allows you to get lightning-fast predictions ⚡ bypassing the tradeoff between accuracy and speed.

                [This notebook](https://colab.research.google.com/drive/1uFCO2UBpH-5l2fk3KmxfU0oupsOC6v2n?authuser=0&pli=1#cell-5=) shows how to generate a pre-trained model and store it in a checkpoint to make it available for public use to forecast new time series never seen by the model. 
                **You can contribute with your pre-trained models by following [this Notebook](https://github.com/Nixtla/transfer-learning-time-series/blob/main/nbs/Transfer_Learning.ipynb) and sending us an email at federico[at]nixtla.io**

                You can also take a look at list of pretrained models here.  Currently we have this ones avaiable in our [API](https://docs.nixtla.io/reference/neural_transfer_neural_transfer_post) or [Demo](http://nixtla.io/transfer-learning/). You can also download the `.ckpt`:
                - [Pretrained N-HiTS M4 Hourly](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nhits_m4_hourly.ckpt)
                - [Pretrained N-HiTS M4 Hourly (Tiny)](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nhits_m4_hourly_tiny.ckpt)
                - [Pretrained N-HiTS M4 Daily](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nhits_m4_daily.ckpt)
                - [Pretrained N-HiTS M4 Monthly](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nhits_m4_monthly.ckpt)
                - [Pretrained N-HiTS M4 Yearly](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nhits_m4_yearly.ckpt)
                - [Pretrained N-BEATS M4 Hourly](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nbeats_m4_hourly.ckpt)
                - [Pretrained N-BEATS M4 Daily](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nbeats_m4_daily.ckpt)
                - [Pretrained N-BEATS M4 Weekly](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nbeats_m4_weekly.ckpt)
                - [Pretrained N-BEATS M4 Monthly](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nbeats_m4_monthly.ckpt)
                - [Pretrained N-BEATS M4 Yearly](https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/nbeats_m4_yearly.ckpt)
                """
            st.write(transfer_text)

        with tab_ref:
            ref_text = """
                If you are interested in the transfer learning literature applied to time series forecasting, take a look at these papers:
                - [Meta-learning framework with applications to zero-shot time-series forecasting](https://arxiv.org/abs/2002.02887)
                - [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://arxiv.org/abs/2201.12886)
            """
            st.write(ref_text)

    with tab_nixtla:
        nixtla_text = """
            Nixtla is a startup that is building forecasting software for Data Scientists and Devs.

            We have been developing different open source libraries for machine learning, statistical and deep learning forecasting.

            In our [GitHub repo](https://github.com/Nixtla), you can find the projects that support this APP.
        """
        st.write(nixtla_text)
        st.image(
            "https://files.readme.io/168cdb2-Screen_Shot_2022-09-30_at_10.40.09.png",
            width=800,
        )

    with st.sidebar:
        st.download_button(
            label="Download historical data as CSV",
            data=convert_df(df),
            file_name="history.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download forecasts as CSV",
            data=convert_df(df_forecast),
            file_name="forecasts.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    st_transfer_learning()
