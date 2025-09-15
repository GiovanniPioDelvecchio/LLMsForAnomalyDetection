import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def extract_ts(symbol, df):
    selected_entries = df.loc[df['symbol'] == symbol].copy()
    # Converte date in datetime
    selected_entries["date"] = pd.to_datetime(selected_entries["date"])
    return selected_entries["close"], selected_entries["date"]


def extract_ts_with_range(symbol, df, start_time=None, end_time=None):
    selected_entries = df.loc[df['symbol'] == symbol].copy()
    # Converte date in datetime
    selected_entries["date"] = pd.to_datetime(selected_entries["date"])

    # Applica filtro sul range se specificato
    if start_time is not None:
        selected_entries = selected_entries[selected_entries["date"] >= pd.to_datetime(start_time)]
    if end_time is not None:
        selected_entries = selected_entries[selected_entries["date"] <= pd.to_datetime(end_time)]

    return selected_entries["close"], selected_entries["date"]


def plot_ts(series_to_plot, dates, symbol):
    plt.figure(figsize=(10, 8), dpi=100)

    # Plot principale
    plt.plot(dates, series_to_plot, label=symbol, color="blue")

    # Evidenzia i NaN
    nan_mask = series_to_plot.isna()
    plt.scatter(dates[nan_mask], [0] * nan_mask.sum(), color="red", marker="x", label="NaN")

    # Formatting asse X (mostra un tick per anno)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"Time Series for {symbol}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(f"{symbol}_series_plot.png")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("./sp225.csv")
    symbol = "AAPL"
    extracted_col, dates = extract_ts(symbol, df)
    print(f"The amount of NaN values for symbol {symbol} is: {sum(extracted_col.isna())}")
    plot_ts(extracted_col, dates, symbol)
