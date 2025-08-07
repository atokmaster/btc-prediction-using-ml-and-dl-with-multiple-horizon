
# Bitcoin Price Prediction Using Machine Learning and Deep Learning

This project presents an AI-based approach to predict the directional movement of Bitcoin (BTC) prices across multiple future time horizons using both Machine Learning (ML) and Deep Learning (DL) models. By combining multi-timeframe OHLCV data with engineered technical features, the system generates classification signals (LONG, SHORT, NO TRADE) to assist in automated crypto trading decisions.

---

## 📌 Project Overview

This repository includes:

- Multi-horizon prediction:
  - 6 Hours (6H), 12 Hours (12H), 1 Day (1D), 3 Days (3D), 7 Days (7D)
- Multi-timeframe input: 1H, 4H, 1D OHLCV data
- Technical indicators: EMA, RSI, MACD, Bollinger Band %, Volume Spike, etc.
- ML models: XGBoost, LightGBM, Random Forest
- DL models: LSTM, BiLSTM, GRU, CNN-1D, CNN-LSTM, Transformer
- Ensemble-based decision logic (LONG / SHORT / NO TRADE)
- Signal output stored in `signal_log.xlsx` and logged historically

---

## 🧠 Best Model Performance (Per Horizon)

| Horizon | Model           | Accuracy | F1 Macro | Source |
|---------|------------------|----------|----------|--------|
| 6H      | XGBOOST_6H       | 0.697    | 0.652    | 1H     |
| 12H     | RANDOMFOREST_12H | 0.770    | 0.772    | 1H     |
| 1D      | LIGHTGBM_1D      | 0.685    | 0.631    | 4H     |
| 3D      | CNN_LSTM64_3D    | 0.761    | 0.799    | 4H     |
| 7D      | GRU64_7D         | 0.623    | 0.603    | 1D     |

> These results are based on validation datasets and early-stopped training with class balancing.

---

## 🗂 Folder Structure

```
├── data/                    # Final feature datasets
├── btc_realtime/           # Latest OHLCV data from Binance API
├── train_model/            # Scripts for model training (e.g. train_6h.py)
├── signal_model/           # Signal prediction & data fetch logic
├── loop_results/           # Model outputs and evaluation (not included in public repo)
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
```

---

## ▶️ How to Run

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run signal generation**
```bash
python signal_model/signal_main.py
```

3. **Check output**
- `signal_log.xlsx` for latest signal
- `signal_log_history.csv` & `.xlsx` for signal logs (private)

---

## 🔐 Notes

This is a **partial public version** of the full BTC signal engine.  
If you're interested in the full version (with all signal logic, model weights, and VPS deployment), feel free to reach out via GitHub or [LinkedIn](https://www.linkedin.com/in/muhammad-fitrah-athaillah-1a2651218/).

---

## 🙋‍♂️ Author

**Muhammad Fitrah Athaillah**  
[LinkedIn](https://www.linkedin.com/in/muhammad-fitrah-athaillah-1a2651218/) • GitHub: `@atokmaster`
