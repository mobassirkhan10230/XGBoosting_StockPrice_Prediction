# 📈 XGBoost Stock Market Prediction

An interactive **Stock Market Prediction Application** built using Python, XGBoost, and Streamlit. The project explores ensemble learning and gradient boosting by applying XGBoost Regression and Classification to historical stock market data.

The application allows users to select a stock, train machine learning models using historical market data, and generate predictions for a selected future date.

## 🎯 Project Objective

The primary objective is to explore how **XGBoost and ensemble learning techniques** can be applied to financial time-series data to estimate future stock prices and predict market direction.

This project focuses on two machine learning tasks:

- **Regression:** Predict the future closing price of a selected stock.
- **Classification:** Predict whether the stock price will move UP or NOT UP.

## 🚀 Key Features

### 1. Historical Stock Data
- Download historical stock market data using Yahoo Finance (`yfinance`).
- Select stocks from a predefined list.
- Choose historical training periods.
- Prepare and clean market data for model training.

### 2. XGBoost Model Training
- Implement XGBoost Regression for future closing-price prediction.
- Implement XGBoost Classification for stock direction prediction.
- Apply hyperparameter optimization using `RandomizedSearchCV`.
- Use `TimeSeriesSplit` for chronological cross-validation.

### 3. Future Stock Prediction
- Select a future prediction date through the Streamlit interface.
- Calculate the corresponding trading-session horizon.
- Generate a predicted future closing price.
- Estimate the probability of an upward price movement.
- Display the predicted stock direction.

### 4. Interactive Web Application
- Built with Streamlit.
- Simple stock selection and training interface.
- Separate model-training and prediction workflows.
- Interactive prediction results.

## 🧠 Machine Learning Concepts Explored

This project provides practical experience with:

- Ensemble Learning
- Gradient Boosting
- Extreme Gradient Boosting (XGBoost)
- Regression and Classification
- Hyperparameter Optimization
- Randomized Search Cross-Validation
- Time-Series Cross-Validation
- Model Evaluation

### Important XGBoost Hyperparameters

| Parameter | Purpose |
|---|---|
| `n_estimators` | Number of boosting trees |
| `max_depth` | Maximum depth of each tree |
| `learning_rate` | Controls the contribution of each tree |
| `subsample` | Fraction of training samples used per tree |
| `colsample_bytree` | Fraction of features used per tree |
| `min_child_weight` | Controls the minimum child-node weight |
| `reg_alpha` | L1 regularization |
| `reg_lambda` | L2 regularization |

## 🛠️ Technology Stack

| Technology | Usage |
|---|---|
| Python | Core programming language |
| XGBoost | Regression and classification |
| Scikit-learn | Model selection and validation |
| Pandas | Data processing |
| NumPy | Numerical operations |
| yfinance | Historical stock market data |
| Joblib | Model serialization |
| Streamlit | Interactive web application |

## ⚙️ Application Workflow

1. Select a stock.
2. Select the historical training period.
3. Download and preprocess historical stock data.
4. Train XGBoost Regression and Classification models.
5. Select a future prediction date.
6. Calculate the trading-session horizon.
7. Generate the predicted closing price and stock direction.
8. Display the results through the Streamlit application.

## 📂 Project Structure

```text
xgboosting/
│
├── data/
│
├── models/
│
├── notebook/
│   ├── app.py
│   ├── data_process.py
│   ├── data_clean.py
│   ├── model.py
│   └── prediction.py
│
├── tools/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd xgboosting
```

### 2. Create a virtual environment

```bash
python -m venv xgboosting
```

Activate the environment:

**Windows:**

```bash
xgboosting\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run notebook/app.py
```

The application will open in your browser.

## 📊 Future Improvements

The project is actively being developed. Planned improvements include:

- Feature engineering using technical indicators such as RSI, MACD, and moving averages.
- Advanced model evaluation using MAE, RMSE, and classification metrics.
- Historical backtesting and baseline comparisons.
- Improved time-series validation.
- Model persistence and model-specific prediction workflows.
- Integration of stock market news and sentiment analysis.

## ⚠️ Disclaimer

This project is developed for educational and experimental purposes. Stock market predictions are inherently uncertain. Model outputs should not be considered financial advice or used as the sole basis for investment decisions.

## 👨‍💻 Author

Developed as a practical Machine Learning project to explore **Ensemble Learning, XGBoost, Hyperparameter Optimization, and Financial Time-Series Prediction**.