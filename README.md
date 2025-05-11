# Keeneland September Yearling Auction Price Predictive Model and Buyer Segmentation Analysis

## Overview
This repository contains a comprehensive analysis of Keeneland September Yearling Auction data from 2020 - 2024, including a predictive model for auction prices and buyer segmentation analysis. The project utilizes machine learning techniques to forecast yearling sale prices based on pedigree, conformation, and market factors, while also segmenting buyers to identify patterns in purchasing behavior.
## Files
- `app.py`: Streamlit app for interacting with the predictive model and viewing predictions.
- `best_lightgbm_mapie_regressor.joblib`: Trained and finetuned LightGBM regressor model with MAPIE for prediction intervals.
- `feature_lists.json`: JSON file containing the feature list used by the model.
- `Keeneland_September_Price_Predictive_Model_training.ipynb`: Jupyter notebook for training the predictive model.
- `requirements.txt`: Python dependencies required to run the project for running the app.py.
- `Buyer_segmentation_analysis.ipynb`: Jupyter notebook analyzing buyer segments analysis.
- `Keeneland_September_Yearling_Auction_Prediction.pdf`: Slide deck summarizing the project, predictive model, and buyer segmentation findings.
- `sold.csv`: Dataset containing Keeneland September Yearling Auction data.

## Setup Instructions
1. Clone this repository: `git clone https://github.com/trungpham25/Keeneland-September-Yearling-Auction-Price-Predictive-Model-and-Buyer-Segmentation-Analysis.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`
4. Access the app at `http://localhost:8501`

## Training Script
- View/run the training script in [Google Colab]([url](https://colab.research.google.com/drive/19FPz0p5FfiIdUrI9LVZeKdGqA8wkvzBt?usp=sharing)).
- Or run locally with Jupyter Notebook: `jupyter notebook Keeneland_September_Price_Predictive_Model_training.ipynb`.

## Buyer Segmentation Analysis
- View/run the script in [Google Colab]([url](https://colab.research.google.com/drive/1Ft8dUbhX5a80yUqGzscpyhdhAGi210e-?usp=sharing))
- See `Buyer_segmentation_analysis.ipynb` for buyer personas and bidding patterns analysis.
- Key findings are summarized in `Keeneland_September_Yearling_Auction_Prediction.pdf`.

## Project Summary
- See `Keeneland_September_Yearling_Auction_Prediction.pdf` for an overview of the project, predictive model (using LightGBM regressor), buyer segmentation, and results.

## Notes
- Ensure Python 3.8+ is installed.
- All files, including the dataset (`sold.csv`), are included in the repository.
- For local execution of notebooks, update file paths if they reference external data (e.g., Google Drive mounts in Colab).
- The trained model (`lightgbm_mapie_regressor.joblib`) is embedded in the Streamlit app and loaded dynamically.


## Disclaimer
This application provides statistical estimates based on historical data. Actual auction prices may vary significantly based on market conditions and individual yearling characteristics.
