import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def get_market_cap_cr(ticker: str) -> float:
    try:
        ticker_obj = yf.Ticker(ticker)
        market_cap = ticker_obj.fast_info.get("market_cap")
        if market_cap is None:
            market_cap = ticker_obj.info.get("marketCap")
        if market_cap is None:
            return np.nan
        return market_cap / 1e7
    except Exception:
        return np.nan


def get_debt_to_equity(ticker: str) -> float:
    try:
        ticker_obj = yf.Ticker(ticker)
        debt_to_equity = ticker_obj.info.get("debtToEquity")
        if debt_to_equity is None:
            return np.nan
        return float(debt_to_equity)
    except Exception:
        return np.nan


def get_company_name(ticker: str) -> str:
    try:
        ticker_obj = yf.Ticker(ticker)
        company_name = ticker_obj.info.get("longName") or ticker_obj.info.get("shortName")
        return company_name if company_name else ticker
    except Exception:
        return ticker


def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def calculate_beta(stock_returns: np.ndarray, index_returns: np.ndarray) -> tuple[float, float]:
    if len(stock_returns) < 2 or len(index_returns) < 2:
        return np.nan, np.nan

    covariance = np.cov(stock_returns, index_returns)[0, 1]
    index_variance = np.var(index_returns, ddof=1)

    if index_variance == 0:
        return np.nan, np.nan

    beta = covariance / index_variance
    correlation = np.corrcoef(stock_returns, index_returns)[0, 1]
    r_squared = correlation**2 if not np.isnan(correlation) else np.nan

    return beta, r_squared


def compute_betas(stocks: list[str], indices: list[str]) -> pd.DataFrame:
    timeframes = [
        ("1y", "1d", "Daily"),
        ("2y", "1wk", "Weekly"),
        ("5y", "1mo", "Monthly"),
    ]

    results = []
    market_cap_cache = {}
    debt_to_equity_cache = {}
    company_name_cache = {}

    for stock in stocks:
        if stock not in market_cap_cache:
            market_cap_cache[stock] = get_market_cap_cr(stock)
        if stock not in debt_to_equity_cache:
            debt_to_equity_cache[stock] = get_debt_to_equity(stock)
        if stock not in company_name_cache:
            company_name_cache[stock] = get_company_name(stock)

        for index in indices:
            betas = {}
            r_squareds = {}

            for period, interval, interval_name in timeframes:
                try:
                    stock_data = download_data(stock, period, interval)
                    index_data = download_data(index, period, interval)

                    if "Close" not in stock_data.columns or "Close" not in index_data.columns:
                        betas[f"Beta_{interval_name}"] = np.nan
                        r_squareds[f"R2_{interval_name}"] = np.nan
                        continue

                    aligned = pd.DataFrame(
                        {
                            "stock": stock_data["Close"],
                            "index": index_data["Close"],
                        }
                    ).dropna()

                    if len(aligned) < 2:
                        betas[f"Beta_{interval_name}"] = np.nan
                        r_squareds[f"R2_{interval_name}"] = np.nan
                        continue

                    stock_returns = aligned["stock"].pct_change().dropna()
                    index_returns = aligned["index"].pct_change().dropna()

                    common_dates = stock_returns.index.intersection(index_returns.index)
                    stock_returns = stock_returns.loc[common_dates]
                    index_returns = index_returns.loc[common_dates]

                    beta, r_squared = calculate_beta(stock_returns.values, index_returns.values)
                    betas[f"Beta_{interval_name}"] = beta
                    r_squareds[f"R2_{interval_name}"] = r_squared
                except Exception:
                    betas[f"Beta_{interval_name}"] = np.nan
                    r_squareds[f"R2_{interval_name}"] = np.nan

            beta_values = [v for v in betas.values() if not np.isnan(v)]
            avg_beta = np.mean(beta_values) if beta_values else np.nan

            r2_values = [v for v in r_squareds.values() if not np.isnan(v)]
            avg_r2 = np.mean(r2_values) if r2_values else np.nan

            result = {
                "Stock": stock,
                "Company_Name": company_name_cache.get(stock, stock),
                "Market_Cap_Cr": market_cap_cache.get(stock, np.nan),
                "Debt_To_Equity": debt_to_equity_cache.get(stock, np.nan),
                "Index": index,
                "Beta_Daily": betas.get("Beta_Daily", np.nan),
                "R2_Daily": r_squareds.get("R2_Daily", np.nan),
                "Beta_Weekly": betas.get("Beta_Weekly", np.nan),
                "R2_Weekly": r_squareds.get("R2_Weekly", np.nan),
                "Beta_Monthly": betas.get("Beta_Monthly", np.nan),
                "R2_Monthly": r_squareds.get("R2_Monthly", np.nan),
                "Beta_Std_Dev": np.std(
                    [
                        betas.get("Beta_Daily", np.nan),
                        betas.get("Beta_Weekly", np.nan),
                        betas.get("Beta_Monthly", np.nan),
                    ]
                ),
                "Beta_Average": avg_beta,
                "R2_Average": avg_r2,
            }
            results.append(result)

    return pd.DataFrame(results)


def add_symbol(session_key: str, symbol: str) -> None:
    clean_symbol = symbol.strip().upper()
    if clean_symbol and clean_symbol not in st.session_state[session_key]:
        st.session_state[session_key].append(clean_symbol)


def remove_symbol(session_key: str, symbol: str) -> None:
    if symbol in st.session_state[session_key]:
        st.session_state[session_key].remove(symbol)


def export_dataframe(df: pd.DataFrame) -> tuple[str, bytes]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Beta_Analysis_{timestamp}.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return filename, csv_bytes


def main() -> None:
    st.set_page_config(page_title="Stock & Index Beta App", layout="wide")
    
    # Hide Streamlit header, footer, and toolbar (including GitHub logo)
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Stock & Index Beta Analysis")
    st.caption("Add stocks and indices, compute beta metrics, and export to CSV.")

    if "stocks" not in st.session_state:
        st.session_state["stocks"] = [""]
    if "indices" not in st.session_state:
        st.session_state["indices"] = ["^NSEI"]
    if "result_df" not in st.session_state:
        st.session_state["result_df"] = None

    left, right = st.columns(2)

    with left:
        st.subheader("Stocks")
        new_stock = st.text_input("Add stock symbol", placeholder="e.g. TCS.NS")
        if st.button("Add Stock", use_container_width=True):
            add_symbol("stocks", new_stock)

        if st.session_state["stocks"]:
            stock_to_remove = st.selectbox("Remove stock", st.session_state["stocks"], key="stock_remove")
            if st.button("Remove Stock", use_container_width=True):
                remove_symbol("stocks", stock_to_remove)

        st.write("Current stocks:")
        st.write(st.session_state["stocks"])

    with right:
        st.subheader("Indices")
        new_index = st.text_input("Add index symbol", placeholder="e.g. ^NSEI")
        if st.button("Add Index", use_container_width=True):
            add_symbol("indices", new_index)

        if st.session_state["indices"]:
            index_to_remove = st.selectbox("Remove index", st.session_state["indices"], key="index_remove")
            if st.button("Remove Index", use_container_width=True):
                remove_symbol("indices", index_to_remove)

        st.write("Current indices:")
        st.write(st.session_state["indices"])

    st.divider()

    if st.button("Run Beta Analysis", type="primary", use_container_width=True):
        if not st.session_state["stocks"]:
            st.error("Please add at least one stock.")
        elif not st.session_state["indices"]:
            st.error("Please add at least one index.")
        else:
            with st.spinner("Fetching data and calculating betas..."):
                st.session_state["result_df"] = compute_betas(
                    st.session_state["stocks"],
                    st.session_state["indices"],
                )

    if st.session_state["result_df"] is not None and not st.session_state["result_df"].empty:
        st.subheader("Results")
        st.dataframe(st.session_state["result_df"], use_container_width=True)

        file_name, csv_bytes = export_dataframe(st.session_state["result_df"])
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=file_name,
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
