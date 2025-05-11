# streamlit_app_lgbm_final_v11.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import time
import os
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import sys
import traceback

try:
    from mapie.regression import MapieRegressor
    from mapie.metrics import regression_coverage_score
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    st.error("Mapie library not found. Please install it (`pip install mapie`) to calculate prediction intervals.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Keeneland September Yearling Auction Price Predictor")

# --- Constants and Configuration ---
MODEL_PATH = "best_lightgbm_mapie_regressor.joblib"
FEATURE_LIST_PATH = "feature_lists.json"
SOLD_DATA_PATH = "sold.csv"
ARTIFACT_DIR = '.'
CURRENT_YEAR = 2025
DEBUG_MODE = False
MAPIE_ALPHA = 0.1 # For 90% prediction interval

# --- Horse Color Mapping ---
COLOR_MAPPING = {'B': 'Bay', 'BAY': 'Bay', 'BL': 'Black', 'BLK': 'Black', 'BLACK': 'Black', 'BR': 'Brown', 'CH': 'Chestnut', 'CHESTNUT': 'Chestnut', 'DB': 'Dark Bay', 'DK B': 'Dark Bay', 'DK BR': 'Dark Brown', 'G': 'Gray/Grey', 'GR': 'Gray/Grey', 'GRAY OR ROAN': 'Gray / Roan', 'RO': 'Roan', 'CH/RO': 'Chestnut Roan', 'DB/BR': 'Dark Bay or Brown', 'DK B/BR': 'Dark Bay or Brown', 'B/BR': 'Bay or Brown', 'Unknown': 'Unknown', 'nan': 'Unknown'}

# --- Helper Function: Clean Name ---
def clean_name_robust(name_str, is_seller=False):
    # ... (Keep the V3 robust cleaning function here) ...
    if pd.isna(name_str): return 'unknown'; name_str = str(name_str).lower().strip(); name_str = re.sub(r'\s*\([\w.]+\)\s*$', '', name_str).strip()
    if is_seller:
        for_pattern = r'[,]?\s+for\s+.*$'; name_str = re.sub(for_pattern, '', name_str, flags=re.IGNORECASE).strip()
        suffixes_list = ['inc', 'llc', 'ltd', 'farm', 'stud', 'agency', 'consignment','agent', 'pinhooker', 'sales', 'bloodstock', 'thoroughbreds', 'equine']; suffixes_pattern = r'[,\s]*\b(' + '|'.join(suffixes_list) + r')[.,]?\s*$'; agent_pattern = r'[,]?\s+(as\s+)?agent(\s+iv|\s+v\b|\s+vi\b|\s+x\b|\s+xi\b|\s+xii\b)?.*?$'; roman_numeral_pattern = r'[,]?\s+[ivxlcdm]+$'
        name_str = re.sub(agent_pattern, '', name_str, flags=re.IGNORECASE).strip(); name_str = re.sub(suffixes_pattern, '', name_str, flags=re.IGNORECASE).strip(); name_str = re.sub(roman_numeral_pattern, '', name_str, flags=re.IGNORECASE).strip(); name_str = re.sub(r'[^\w\s]+$', '', name_str).strip()
    name_str = re.sub(r'\s+', ' ', name_str).strip()
    return name_str if name_str else 'unknown'

# --- Helper function to display more readable horse colors ---
def get_readable_color(code):
    return COLOR_MAPPING.get(str(code).strip().upper(), str(code).strip())

# --- Load Model Function ---
@st.cache_resource
def load_model(path):
    try: model = joblib.load(path); print(f"Model object '{os.path.basename(path)}' loaded."); return model
    except FileNotFoundError: st.error(f"Model file not found: {path}"); return None
    except Exception as e: st.error(f"Error loading model: {e}"); return None

# --- Load Base Data and Preprocess ---
@st.cache_data
def load_and_prep_base_data(path):
    # ... (Keep V4 load_and_prep_base_data) ...
    print("Loading and prepping base data...");
    try:
        try: df = pd.read_csv(path)
        except UnicodeDecodeError: df = pd.read_csv(path, encoding='latin1')
        df.columns = df.columns.str.strip(); required_base_cols = ['Sire', 'Dam', 'PropertyLine1', 'Year', 'Price', 'Session', 'Color', 'Sex', 'Brilliant', 'Intermediate', 'Classic', 'Solid', 'Professional']; missing = [col for col in required_base_cols if col not in df.columns];
        if missing: raise ValueError(f"Base data missing required columns: {missing}")
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce'); df['Price_Str'] = df['Price'].astype(str).str.replace(r'[$,]', '', regex=True); df['Price'] = pd.to_numeric(df['Price_Str'], errors='coerce'); df.drop(columns=['Price_Str'], inplace=True)
        price_median = df['Price'].median(); nan_price_count = df['Price'].isnull().sum();
        if nan_price_count > 0: df['Price'] = df['Price'].fillna(price_median)
        df['Log_Price'] = np.log1p(df['Price'].clip(lower=1)); df['Session'] = pd.to_numeric(df['Session'], errors='coerce');
        df['Sire_Clean'] = df['Sire'].apply(lambda x: clean_name_robust(x, is_seller=False)); df['PropertyLine1_Clean'] = df['PropertyLine1'].apply(lambda x: clean_name_robust(x, is_seller=True))
        if 'Color' in df.columns: df['Color'] = df['Color'].astype(str).fillna('Unknown')
        if 'Sex' in df.columns: df['Sex'] = df['Sex'].astype(str).fillna('Unknown')
        numeric_base_cols = ['Sire_starters', 'Sire_winners', 'Sire_BW', 'Sire_earnings', 'Sire_aei','BS_foals', 'BS_starters', 'BS_winners', 'BS_BW', 'BS_earnings', 'BS_aei','Brilliant', 'Intermediate', 'Classic', 'Solid', 'Professional','Dosage_Index', 'Center_of_Distribution', 'Hip']
        for col in numeric_base_cols:
             if col in df.columns:
                  if isinstance(df[col].dtype, object) and 'earnings' in col.lower(): df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                  df[col] = pd.to_numeric(df[col], errors='coerce')
             else: df[col] = np.nan
        if all(c in df.columns for c in ['Brilliant', 'Intermediate', 'Classic', 'Solid', 'Professional']):
             b, i, c, s, p = pd.to_numeric(df['Brilliant'],errors='coerce').fillna(0), pd.to_numeric(df['Intermediate'],errors='coerce').fillna(0), pd.to_numeric(df['Classic'],errors='coerce').fillna(0), pd.to_numeric(df['Solid'],errors='coerce').fillna(0), pd.to_numeric(df['Professional'],errors='coerce').fillna(0)
             di_denominator = (s + p) + (c / 2.0); cd_denominator = b + i + c + s + p
             df['DI_Calc'] = np.where( di_denominator != 0, ((b + i) + (c / 2.0)) / di_denominator, 3.0)
             df['CD_Calc'] = np.where(cd_denominator != 0, ((b - s) + (i - p)) / cd_denominator, 0.0)
             if 'Dosage_Index' not in df.columns: df['Dosage_Index'] = np.nan
             if 'Center_of_Distribution' not in df.columns: df['Center_of_Distribution'] = np.nan
             df['Dosage_Index'] = df['Dosage_Index'].fillna(df['DI_Calc'])
             df['Center_of_Distribution'] = df['Center_of_Distribution'].fillna(df['CD_Calc'])
        print("Base data loaded and prepped."); return df
    except FileNotFoundError: st.error(f"Base data file '{path}' not found."); return None
    except Exception as e: st.error(f"Error loading/prepping data: {e}"); return None


# --- On-The-Fly Calculation Functions ---
@st.cache_data
def calculate_sire_lookups(_df_base):
    print("Calculating Sire lookups...");
    try: df_sire = _df_base[['Sire_Clean', 'Price', 'Log_Price', 'Year']].copy(); df_sire['Year_Num'] = pd.to_numeric(df_sire['Year'], errors='coerce').fillna(df_sire['Year'].median()).astype(int); min_offspring_threshold=5; elite_percentile=90; w_p90=0.4; w_elite=0.3; w_median=0.2; w_p25=0.1; epsilon_div=1e-6; elite_threshold_price=np.percentile(df_sire['Price'].dropna(), elite_percentile); global_median_p25=df_sire['Log_Price'].dropna().quantile(0.25); global_median_p90=df_sire['Log_Price'].dropna().quantile(0.90); global_median_log_price=df_sire['Log_Price'].median(); sire_stats_df = df_sire.groupby('Sire_Clean').agg(count=('Sire_Clean', 'size'), median_log_price=('Log_Price', 'median'), p25_log_price=('Log_Price', lambda x: x.quantile(0.25)), p90_log_price=('Log_Price', lambda x: x.quantile(0.90)), elite_count=('Price', lambda x: (x > elite_threshold_price).sum()), first_year=('Year_Num', 'min')).reset_index(); sire_stats_df['elite_rate'] = sire_stats_df['elite_count'] / (sire_stats_df['count'] + epsilon_div); sire_stats_df['median_log_price_rank'] = sire_stats_df['median_log_price'].rank(pct=True); sire_stats_df.fillna({'p25_log_price': global_median_p25, 'p90_log_price': global_median_p90, 'elite_rate': 0.0, 'median_log_price': global_median_log_price, 'median_log_price_rank': 0.5}, inplace=True); is_low_volume = sire_stats_df['count'] < min_offspring_threshold; sire_stats_df.loc[is_low_volume, ['p25_log_price', 'p90_log_price', 'elite_rate', 'median_log_price_rank']] = [global_median_p25, global_median_p90, 0.0, 0.5]; scaler_sire = MinMaxScaler(); metrics_to_scale = ['p90_log_price', 'elite_rate', 'median_log_price_rank', 'p25_log_price']; scaled_metric_names = [f"{col}_scaled" for col in metrics_to_scale]; sire_stats_df[scaled_metric_names] = scaler_sire.fit_transform(sire_stats_df[metrics_to_scale]); sire_stats_df['Sire_Reputation'] = (w_p90 * sire_stats_df['p90_log_price_scaled'] + w_elite * sire_stats_df['elite_rate_scaled'] + w_median * sire_stats_df['median_log_price_rank_scaled'] + w_p25 * sire_stats_df['p25_log_price_scaled']); sire_rep_dict = pd.Series(sire_stats_df['Sire_Reputation'].values, index=sire_stats_df['Sire_Clean']).to_dict(); sire_year_dict = pd.Series(sire_stats_df['first_year'].values.astype(int), index=sire_stats_df['Sire_Clean']).to_dict(); sire_rep_median_val = sire_stats_df['Sire_Reputation'].median(); return sire_rep_dict, sire_year_dict, sire_rep_median_val
    except Exception as e: st.error(f"Error calculating Sire lookups: {e}"); return {}, {}, 0.5

@st.cache_data
def calculate_seller_lookups(_df_base):
    print("Calculating Seller lookups...");
    if 'PropertyLine1_Clean' not in _df_base.columns: st.error("Seller lookup error: 'PropertyLine1_Clean' missing."); return {}, 0.5
    try: df_seller = _df_base[['PropertyLine1_Clean', 'Price', 'Log_Price']].copy(); min_sales_threshold=3; seller_elite_percentile=95; w_p75=0.35; w_elite=0.30; w_median_rank=0.20; w_volume=0.15; epsilon_div=1e-6; elite_thresh = np.percentile(df_seller['Price'].dropna(), seller_elite_percentile); global_p75 = df_seller['Log_Price'].quantile(0.75); global_median = df_seller['Log_Price'].median(); seller_stats = df_seller.groupby('PropertyLine1_Clean').agg(vol=('PropertyLine1_Clean', 'size'), med=('Log_Price', 'median'), p75=('Log_Price', lambda x: x.quantile(0.75)), elite=('Price', lambda x: (x > elite_thresh).sum())).reset_index(); seller_stats['elite_rate'] = seller_stats['elite'] / (seller_stats['vol'] + epsilon_div); seller_stats['med_rank'] = seller_stats['med'].rank(pct=True); seller_stats['vol_rank'] = seller_stats['vol'].rank(pct=True); seller_stats.fillna({'p75': global_p75, 'elite_rate': 0.0, 'med': global_median, 'med_rank': 0.5, 'vol_rank': 0.5}, inplace=True); low = seller_stats['vol'] < min_sales_threshold; seller_stats.loc[low, ['p75', 'elite_rate', 'med_rank', 'vol_rank']] = [global_p75, 0.0, 0.5, 0.5]; scaler = MinMaxScaler(); metrics = ['p75', 'elite_rate', 'med_rank', 'vol_rank']; seller_stats[[f"{m}_sc" for m in metrics]] = scaler.fit_transform(seller_stats[metrics]); seller_stats['Seller_Reputation'] = (w_p75 * seller_stats['p75_sc'] + w_elite * seller_stats['elite_rate_sc'] + w_median_rank * seller_stats['med_rank_sc'] + w_volume * seller_stats['vol_rank_sc']); seller_rep_dict = pd.Series(seller_stats['Seller_Reputation'].values, index=seller_stats['PropertyLine1_Clean']).to_dict(); seller_rep_median_val = seller_stats['Seller_Reputation'].median(); return seller_rep_dict, seller_rep_median_val
    except Exception as e: st.error(f"Error calculating Seller lookups: {e}"); return {}, 0.5

@st.cache_data
def calculate_dam_lookups(_df_base):
    dam_key_col = 'Dam'; print(f"Calculating Dam lookups using column: {dam_key_col}...");
    try: df_dam = _df_base[[dam_key_col, 'Price', 'Log_Price']].copy(); min_offs=2; elite_pct=90; w_p75=0.35; w_elite=0.30; w_med=0.20; w_vol=0.15; epsilon_div=1e-6; elite_thresh = np.percentile(df_dam['Price'].dropna(), elite_pct); global_p75 = df_dam['Log_Price'].quantile(0.75); global_median = df_dam['Log_Price'].median(); dam_stats = df_dam.groupby(dam_key_col).agg( vol=(dam_key_col, 'size'), med=('Log_Price', 'median'), p75=('Log_Price', lambda x: x.quantile(0.75)), elite=('Price', lambda x: (x > elite_thresh).sum())).reset_index(); dam_stats['elite_rate'] = dam_stats['elite'] / (dam_stats['vol'] + epsilon_div); dam_stats['med_rank'] = dam_stats['med'].rank(pct=True); dam_stats['vol_rank'] = dam_stats['vol'].rank(pct=True); dam_stats.fillna({'p75': global_p75, 'elite_rate': 0.0, 'med': global_median, 'med_rank': 0.5, 'vol_rank': 0.5}, inplace=True); low = dam_stats['vol'] < min_offs; dam_stats.loc[low, ['p75', 'elite_rate', 'med_rank', 'vol_rank']] = [global_p75, 0.0, 0.5, 0.5]; scaler = MinMaxScaler(); metrics = ['p75', 'elite_rate', 'med_rank', 'vol_rank']; dam_stats[[f"{m}_sc" for m in metrics]] = scaler.fit_transform(dam_stats[metrics]); dam_stats['Dam_Reputation'] = (w_p75 * dam_stats['p75_sc'] + w_elite * dam_stats['elite_rate_sc'] + w_med * dam_stats['med_rank_sc'] + w_vol * dam_stats['vol_rank_sc']); dam_rep_dict = pd.Series(dam_stats['Dam_Reputation'].values, index=dam_stats[dam_key_col]).to_dict(); dam_rep_median_val = dam_stats['Dam_Reputation'].median(); return dam_rep_dict, dam_rep_median_val
    except Exception as e: st.error(f"Error calculating Dam lookups: {e}"); return {}, 0.5

@st.cache_data
def calculate_imputation_medians(_df_base, _numerical_features):
    print("Calculating imputation medians..."); medians = {}
    cols_for_median_base = ['Session', 'Hip', 'Brilliant', 'Intermediate', 'Classic', 'Solid', 'Professional','Sire_starters', 'Sire_winners', 'Sire_BW', 'Sire_earnings', 'Sire_aei','BS_foals', 'BS_starters', 'BS_winners', 'BS_BW', 'BS_earnings', 'BS_aei','Dosage_Index', 'Center_of_Distribution']
    all_numeric_needed = list(set(cols_for_median_base + _numerical_features))
    for col in all_numeric_needed:
        if col in _df_base.columns and pd.api.types.is_numeric_dtype(_df_base[col].dtype): median_val = _df_base[col].median(); median_val = median_val if not pd.isna(median_val) else (1.0 if 'aei' in col.lower() else 0); medians[col] = float(median_val)
        else: medians[col] = 1.0 if 'aei' in col.lower() else 0.0
    if 'Session' in _df_base.columns and pd.api.types.is_numeric_dtype(_df_base['Session']): medians['max_session_train'] = int(_df_base['Session'].max())
    else: medians['max_session_train'] = 12
    print(f"Calculated {len(medians)} median values."); return medians

@st.cache_data
def create_sire_stats_dict(_df_base):
    print("Creating sire statistics dictionary for auto-fill..."); sire_stats = {}
    try:
        stat_cols = ['Sire_starters', 'Sire_winners', 'Sire_BW', 'Sire_earnings', 'Sire_aei', 'Year']; required_cols = ['Sire'] + stat_cols
        if not all(col in _df_base.columns for col in required_cols): return {}
        df_copy = _df_base[required_cols].copy();
        for col in stat_cols:
             if col == 'Year': df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce'); year_median = df_copy[col].median(); df_copy[col] = df_copy[col].fillna(year_median if not pd.isna(year_median) else 2022).astype(int)
             elif col != 'Sire': numeric_series = pd.to_numeric(df_copy[col], errors='coerce'); df_copy[col] = numeric_series.fillna(0)
        df_copy.dropna(subset=['Sire', 'Year'], inplace=True); df_copy.sort_values(by=['Sire', 'Year'], ascending=[True, False], inplace=True); latest_sire_data = df_copy.drop_duplicates(subset=['Sire'], keep='first')
        for _, row in latest_sire_data.iterrows():
            sire_name_key = row['Sire']
            sire_stats[sire_name_key] = {'Sire_starters': int(row.get('Sire_starters', 0)), 'Sire_winners': int(row.get('Sire_winners', 0)), 'Sire_BW': int(row.get('Sire_BW', 0)), 'Sire_earnings': int(row.get('Sire_earnings', 0)), 'Sire_aei': float(row.get('Sire_aei', 1.0))}
        print(f"Created sire_stats_dict with {len(sire_stats)} entries."); return sire_stats
    except Exception as e: st.error(f"Error creating sire stats dict: {e}"); return {}

# --- Define handle_sire_change callback ---
def handle_sire_change():
    selected_sire = st.session_state.get('selected_sire', "")
    sire_stats_dict = st.session_state.get('sire_stats_dict', {})
    if selected_sire and selected_sire in sire_stats_dict: st.session_state.current_sire_stats = sire_stats_dict[selected_sire]
    else: st.session_state.current_sire_stats = {}
    if DEBUG_MODE: print(f"handle_sire_change: Sire='{selected_sire}' -> Updated current_sire_stats state.")

# --- Load Model and Feature Configuration ---
try:
    if not os.path.isdir(ARTIFACT_DIR): ARTIFACT_DIR = '.'
    feature_list_full_path = os.path.join(ARTIFACT_DIR, FEATURE_LIST_PATH); model_full_path = os.path.join(ARTIFACT_DIR, MODEL_PATH); sold_data_full_path = os.path.join(ARTIFACT_DIR, SOLD_DATA_PATH)
    with open(feature_list_full_path, 'r') as f: feature_config = json.load(f)
    FEATURES_TO_USE = feature_config.get('features_to_use_selected', feature_config.get('features_to_use', [])); NUMERICAL_FEATURES = feature_config.get('numerical_features_selected', feature_config.get('numerical_features', [])); CATEGORICAL_FEATURES = feature_config.get('categorical_features_selected', feature_config.get('categorical_features', []))
    if not FEATURES_TO_USE: raise ValueError("Feature list empty.")
    st.sidebar.success("Feature configuration loaded.")
except Exception as e: st.error(f"Error loading feature config: {e}"); st.stop()

# --- Initialize Session State & Load Data/Model/Lookups ONCE ---
if 'init_done' not in st.session_state:
    st.session_state.init_done = False; print("Performing one-time initialization...")
    with st.spinner("Application initializing... Please wait."):
        base_data = load_and_prep_base_data(sold_data_full_path); model = load_model(model_full_path)
        if base_data is not None and model is not None and FEATURES_TO_USE and NUMERICAL_FEATURES:
            st.session_state.sold_df_base = base_data; st.session_state.model_pipeline = model
            s_rep_lkp, s_yr_lkp, s_rep_med = calculate_sire_lookups(base_data); sell_rep_lkp, sell_rep_med = calculate_seller_lookups(base_data); d_rep_lkp, d_rep_med = calculate_dam_lookups(base_data); imp_med = calculate_imputation_medians(base_data, NUMERICAL_FEATURES); sire_stats = create_sire_stats_dict(base_data)
            st.session_state.sire_rep_lookup = s_rep_lkp; st.session_state.sire_year_lookup = s_yr_lkp; st.session_state.sire_rep_median = s_rep_med; st.session_state.seller_rep_lookup = sell_rep_lkp; st.session_state.seller_rep_median = sell_rep_med; st.session_state.dam_rep_lookup = d_rep_lkp; st.session_state.dam_rep_median = d_rep_med; st.session_state.imputation_medians = imp_med; st.session_state.sire_stats_dict = sire_stats
            if 'selected_sire' not in st.session_state: st.session_state.selected_sire = ""
            if 'current_sire_stats' not in st.session_state: st.session_state.current_sire_stats = {}
            if 'prediction_result' not in st.session_state: st.session_state.prediction_result = None
            st.session_state.init_done = True; print("Initialization complete."); st.rerun()
        else: st.error("Initialization failed: Could not load data/model or calculate lookups."); st.stop()

# --- Feature Engineering Function ---
def calculate_features(input_data, features_to_use, numerical_features, categorical_features, imp_medians, sire_rep_lkp, sire_year_lkp, sire_rep_med, seller_rep_lkp, seller_rep_med, dam_rep_lkp, dam_rep_med):
    """Applies ALL feature engineering steps based on input dict."""

    if DEBUG_MODE: print("--- Entering calculate_features ---"); start_time_fe_internal = time.time()
    processed_input = input_data.copy(); epsilon_div = 1e-6; all_expected_inputs = list(imp_medians.keys()) + ['Sire', 'Dam', 'PropertyLine1', 'Color', 'Sex'];
    for col in all_expected_inputs:
        is_needed = col in features_to_use or col in ['Sire_starters', 'BS_starters', 'Brilliant', 'Intermediate', 'Classic', 'Solid', 'Professional', 'Session', 'Sire_aei', 'BS_aei', 'Sire_BW', 'BS_BW', 'Year', 'Sex']
        if is_needed:
             input_val = processed_input.get(col); default_val = imp_medians.get(col); is_numeric_col = col in imp_medians and isinstance(default_val, (int, float, np.number)); is_missing_or_nan = (input_val is None or pd.isna(input_val) or str(input_val).strip() == "")
             if is_missing_or_nan:
                 if pd.isna(default_val): default_val = 1.0 if 'aei' in col.lower() else 0.0 if is_numeric_col else "Unknown"
                 processed_input[col] = default_val
             elif is_numeric_col:
                 numeric_val = pd.to_numeric(input_val, errors='coerce');
                 if pd.isna(numeric_val): processed_input[col] = default_val
                 else: processed_input[col] = float(numeric_val) if isinstance(default_val, (float, np.floating)) else int(numeric_val)
             else: processed_input[col] = str(input_val)
    processed_input['Year'] = CURRENT_YEAR
    try: df_eng = pd.DataFrame([processed_input])
    except Exception as df_create_error: st.error(f"Error creating DataFrame: {df_create_error}"); raise
    try:
        df_eng['Sire_Clean'] = clean_name_robust(df_eng['Sire'].iloc[0], is_seller=False); df_eng['Seller_Clean'] = clean_name_robust(df_eng['PropertyLine1'].iloc[0], is_seller=True); dam_lookup_key_value = df_eng['Dam'].iloc[0]
        df_eng['Sire_Reputation'] = sire_rep_lkp.get(df_eng['Sire_Clean'].iloc[0], sire_rep_med); df_eng['Seller_Reputation'] = seller_rep_lkp.get(df_eng['Seller_Clean'].iloc[0], seller_rep_med); df_eng['Dam_Reputation'] = dam_rep_lkp.get(dam_lookup_key_value, dam_rep_med)
        sire_first_yr = sire_year_lkp.get(df_eng['Sire_Clean'].iloc[0], None); sire_start_val = df_eng['Sire_starters'].iloc[0]; is_first_yr = (sire_first_yr is not None) and (df_eng['Year'].iloc[0] == sire_first_yr); is_few_start = (sire_start_val <= 5); df_eng['Is_True_First_Crop'] = 1 if (is_first_yr and is_few_start) else 0
    except Exception as lookup_error: st.error(f"Error during lookups: {lookup_error}"); raise
    try:
        rate_inputs = ['Sire_starters', 'Sire_winners', 'Sire_BW', 'Sire_earnings', 'BS_starters', 'BS_winners', 'BS_BW', 'BS_earnings']; # Ensure these cols are numeric 0-filled
        for col in rate_inputs: df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce').fillna(0)
        sire_starters_safe = max(df_eng['Sire_starters'].iloc[0], epsilon_div); bs_starters_safe = max(df_eng['BS_starters'].iloc[0], epsilon_div)
        df_eng['Sire_win_rate'] = df_eng['Sire_winners'].iloc[0] / sire_starters_safe; df_eng['Sire_BW_rate'] = df_eng['Sire_BW'].iloc[0] / sire_starters_safe; df_eng['BS_win_rate'] = df_eng['BS_winners'].iloc[0] / bs_starters_safe; df_eng['BS_BW_rate'] = df_eng['BS_BW'].iloc[0] / bs_starters_safe; df_eng['Sire_earnings_per_starter'] = df_eng['Sire_earnings'].iloc[0] / sire_starters_safe; df_eng['BS_earnings_per_starter'] = df_eng['BS_earnings'].iloc[0] / bs_starters_safe
        dosage_cols = ['Brilliant', 'Intermediate', 'Classic', 'Solid', 'Professional']; [df_eng[col].fillna(0, inplace=True) for col in dosage_cols if col in df_eng]; df_eng['Total_dosage_points'] = df_eng[dosage_cols].sum(axis=1).iloc[0]; total_dosage_points_safe = df_eng['Total_dosage_points'] + epsilon_div;
        b, i, c, s, p = df_eng['Brilliant'].iloc[0], df_eng['Intermediate'].iloc[0], df_eng['Classic'].iloc[0], df_eng['Solid'].iloc[0], df_eng['Professional'].iloc[0] # Extract scalars
        di_denominator = (s + p) + (c / 2.0); cd_denominator = b + i + c + s + p
        calc_di = ((b + i) + (c / 2.0)) / max(di_denominator, epsilon_div) if di_denominator != 0 else 3.0
        calc_cd = ((b - s) + (i - p)) / max(cd_denominator, epsilon_div) if cd_denominator != 0 else 0.0
        if 'Dosage_Index' in features_to_use: df_eng['Dosage_Index'] = calc_di # Assign calculated
        if 'Center_of_Distribution' in features_to_use: df_eng['Center_of_Distribution'] = calc_cd # Assign calculated
        df_eng['Speed_score'] = (b*2 + i) / total_dosage_points_safe; df_eng['Stamina_score'] = (p*2 + s) / total_dosage_points_safe; df_eng['Balance_score'] = abs(df_eng['Speed_score'] - df_eng['Stamina_score']); df_eng['Classic_emphasis'] = c / total_dosage_points_safe; df_eng['Dosage_profile_completeness'] = (df_eng['Total_dosage_points'] > 0).astype(int); [df_eng[col].fillna(0, inplace=True) for col in ['Speed_score', 'Stamina_score', 'Balance_score', 'Classic_emphasis']]
        df_eng['Sire_aei'] = pd.to_numeric(df_eng.get('Sire_aei'), errors='coerce').fillna(1.0); df_eng['BS_aei'] = pd.to_numeric(df_eng.get('BS_aei'), errors='coerce').fillna(1.0); df_eng['Pedigree_quality_score'] = ((df_eng['Sire_aei'] - 1) * 1.5 + (df_eng['BS_aei'] - 1) * 1.0).iloc[0]; df_eng['Sire_BS_aei_interaction'] = (df_eng['Sire_aei'] * df_eng['BS_aei']).iloc[0]; df_eng['Sire_BS_BW_interaction'] = (df_eng['Sire_BW'] * df_eng['BS_BW']).iloc[0]
        df_eng['Session_quality'] = np.exp(-0.2 * (df_eng['Session'] - 1)).iloc[0]
        if 'Session_category' in features_to_use: max_session_train = imp_medians.get('max_session_train', 12); bins = [0, 2, 4, 7, 10, max(max_session_train, 11) + 1]; labels = ['S1-2','S3-4','S5-7','S8-10','S11+']; df_eng['Session_category'] = pd.cut(df_eng['Session'], bins=bins, labels=labels[:len(labels) if max_session_train>=11 else pd.cut([max_session_train],bins=bins,right=True,include_lowest=True).codes[0]+1], right=True, include_lowest=True).astype(str).fillna('Unknown').iloc[0] # Robust cut
        median_sire_aei = imp_medians.get('Sire_aei', 1.0); median_bs_aei = imp_medians.get('BS_aei', 1.0); df_eng['Sire_aei_relative'] = df_eng['Sire_aei'].iloc[0] / max(median_sire_aei, epsilon_div); df_eng['BS_aei_relative'] = df_eng['BS_aei'].iloc[0] / max(median_bs_aei, epsilon_div)
        df_eng['Inbreeding_proxy'] = (df_eng['Sire_Clean'] == dam_lookup_key_value).astype(int).iloc[0]; df_eng['Is_male'] = 1 if str(input_data.get('Sex', 'Unknown')).upper() in ['C', 'COLT'] else 0
        df_eng['SireRep_x_SessionQuality'] = df_eng['Sire_Reputation'] * df_eng['Session_quality']; df_eng['SellerRep_x_SessionQuality'] = df_eng['Seller_Reputation'] * df_eng['Session_quality']; df_eng['SireRep_x_DamRep'] = df_eng['Sire_Reputation'] * df_eng['Dam_Reputation']; df_eng['Pedigree_x_SessionQuality'] = df_eng['Pedigree_quality_score'] * df_eng['Session_quality']
    except Exception as fe_error: st.error(f"Error during adv FE calc: {fe_error}"); raise
    for col in categorical_features:
        if col in df_eng.columns: df_eng[col] = df_eng[col].astype(str)
    if 'Hip' in features_to_use and 'Hip' in df_eng.columns: df_eng['Hip'] = pd.to_numeric(df_eng['Hip'], errors='coerce').fillna(-1).astype(int).astype(str).replace('-1', 'Unknown')
    final_df = pd.DataFrame(index=[0]); missing_fe_cols = []
    for feature in features_to_use:
        if feature in df_eng.columns: value = df_eng[feature].iloc[0]
        else: value = imp_medians.get(feature, 0 if feature in numerical_features else "Unknown"); missing_fe_cols.append(feature)
        if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)): value = imp_medians.get(feature, 0)
        elif pd.isna(value): value = "Unknown" if feature in categorical_features else imp_medians.get(feature, 0)
        final_df[feature] = value
    if missing_fe_cols: print(f"Warning: Features missing from FE df: {missing_fe_cols}. Used defaults.")
    final_df = final_df[features_to_use] # Ensure order
    if DEBUG_MODE: print(f"FE Internal Time: {time.time() - start_time_fe_internal:.3f}s");
    return final_df


# --- Streamlit UI ---
# (Apply Styling from V9)
st.markdown(""" <style> .stMetric > label {font-size: 1.1rem;} .stMetric > div {font-size: 1.3rem;} </style> """, unsafe_allow_html=True)
st.markdown(f"<div style='background: linear-gradient(90deg, #1e5631 0%, #2e7d32 100%); padding: 15px; border-radius: 8px; margin-bottom: 25px;'><h1 style='color: white; text-align: center; margin:0;'>Keeneland September Yearling Auction Price Predictor</h1></div>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 25px;'>Enter yearling details for the estimated price of {CURRENT_YEAR} auction.</p>", unsafe_allow_html=True)
st.sidebar.header("About"); st.sidebar.info(f"Uses the {os.path.basename(MODEL_PATH).split('_')[1].upper()} Mapie model to predict horse prices based on historical data.")

if not st.session_state.get('init_done', False): st.warning("Initializing... Please wait."); st.stop()

# Retrieve from session state
sold_df_base = st.session_state.sold_df_base; imputation_medians = st.session_state.imputation_medians; sire_stats_dict = st.session_state.sire_stats_dict; model_pipeline = st.session_state.model_pipeline
sire_rep_lookup = st.session_state.sire_rep_lookup; sire_year_lookup = st.session_state.sire_year_lookup; sire_rep_median = st.session_state.sire_rep_median
seller_rep_lookup = st.session_state.seller_rep_lookup; seller_rep_median = st.session_state.seller_rep_median
dam_rep_lookup = st.session_state.dam_rep_lookup; dam_rep_median = st.session_state.dam_rep_median
if not sire_rep_lookup or not seller_rep_lookup or not dam_rep_lookup: st.error("Lookup calculation failed."); st.stop()
if not imputation_medians: st.error("Median calculation failed."); st.stop()


# --- Input Form ---
# Define options
sire_options = [""] + sorted(sold_df_base['Sire'].dropna().unique().tolist())
dam_options = [""] + sorted(sold_df_base['Dam'].dropna().unique().tolist())
seller_options = ["", "Unknown"] + sorted(sold_df_base['PropertyLine1'].dropna().unique().tolist())
raw_color_options = sorted(sold_df_base['Color'].astype(str).fillna('Unknown').unique())
color_display_values = [(get_readable_color(c), c) for c in raw_color_options]
color_display_list = sorted([disp for disp, val in color_display_values])
default_color_display = get_readable_color('DB/BR'); default_color_index = color_display_list.index(default_color_display) if default_color_display in color_display_list else 0
sex_options = sorted(sold_df_base['Sex'].astype(str).fillna('Unknown').unique())

# Sire selection outside the form
st.markdown("##### Select Sire*")
st.selectbox("Select Sire*", options=sire_options, key="selected_sire", on_change=handle_sire_change, label_visibility="collapsed")

with st.form("yearling_form"):
    st.markdown("#### Yearling Information (* = Required)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Pedigree & Sale")
        st.markdown(f"**Selected Sire:** {st.session_state.selected_sire or 'None'}")
        dam_name = st.selectbox("Dam Name*", options=dam_options, index=0, key="dam_select_widget")
        seller_name = st.selectbox("Seller/Consignor*", options=seller_options, index=0, key="seller_select_widget")
        session = st.number_input("Session Number*", min_value=1, value=int(imputation_medians.get('Session', 5)), step=1)
        selected_color_display = st.selectbox("Color", options=color_display_list, index=default_color_index)
        horse_color_code = next((value for display, value in color_display_values if display == selected_color_display), selected_color_display)
        sex = st.selectbox("Sex", options=sex_options, index=sex_options.index('Colt') if 'Colt' in sex_options else 0)
        if 'Hip' in FEATURES_TO_USE: hip = st.number_input("Hip Number", min_value=0, value=int(imputation_medians.get('Hip', 1500)), step=1, help="Enter 0 if unknown")

        # *** Dosage Section Moved to col1 ***
        with st.container(): # Use container instead of expander
             st.markdown("##### Dosage Profile*")
             st.caption("Enter Dosage points")
             dos_col1, dos_col2 = st.columns(2) # Inner columns for dosage
             with dos_col1:
                 brill = st.number_input("Brilliant*", min_value=0, value=int(imputation_medians.get('Brilliant', 3)), step=1)
                 inter = st.number_input("Intermediate*", min_value=0, value=int(imputation_medians.get('Intermediate', 5)), step=1)
                 clas = st.number_input("Classic*", min_value=0, value=int(imputation_medians.get('Classic', 8)), step=1)
             with dos_col2:
                 sol = st.number_input("Solid*", min_value=0, value=int(imputation_medians.get('Solid', 1)), step=1)
                 prof = st.number_input("Professional*", min_value=0, value=int(imputation_medians.get('Professional', 0)), step=1)
                 # DI/CD Inputs removed


    with col2:
        st.markdown("##### Sire Statistics")
        sire_defaults = st.session_state.current_sire_stats; use_median_defaults = not sire_defaults
        sire_starters_init = int(sire_defaults.get('Sire_starters') if not use_median_defaults else imputation_medians.get('Sire_starters', 100)); sire_winners_init = int(sire_defaults.get('Sire_winners') if not use_median_defaults else imputation_medians.get('Sire_winners', 60)); sire_bw_init = int(sire_defaults.get('Sire_BW') if not use_median_defaults else imputation_medians.get('Sire_BW', 5)); sire_earnings_init = int(sire_defaults.get('Sire_earnings') if not use_median_defaults else imputation_medians.get('Sire_earnings', 10000000)); sire_aei_init = float(sire_defaults.get('Sire_aei') if not use_median_defaults else imputation_medians.get('Sire_aei', 1.20))
        sire_starters = st.number_input("Sire's Foals Starters", min_value=0, value=sire_starters_init, step=10)
        sire_winners = st.number_input("Sire's Foals Winners", min_value=0, value=sire_winners_init, step=5)
        sire_bw = st.number_input("Sire's Foals Black-Type Winners", min_value=0, value=sire_bw_init, step=1)
        sire_earnings = st.number_input("Sire's Foals Earnings ($)", min_value=0, value=sire_earnings_init, step=1000000, format="%d")
        sire_aei = st.number_input("Sire's Foals AEI", min_value=0.0, max_value=10.0, value=sire_aei_init, step=0.05, format="%.2f")
        if st.session_state.selected_sire and st.session_state.selected_sire in st.session_state.sire_stats_dict: st.caption("Stats auto-filled from database.")
        else: st.caption("Using default median stats.")
        st.divider(); st.markdown("##### Broodmare Sire (BS) Statistics")
        bs_foals = st.number_input("BS Foals", min_value=0, value=int(imputation_medians.get('BS_foals', 100)), step=10); bs_starters = st.number_input("BS's Foals Starters", min_value=0, value=int(imputation_medians.get('BS_starters', 60)), step=10); bs_winners = st.number_input("BS's Foals Winners", min_value=0, value=int(imputation_medians.get('BS_winners', 40)), step=5); bs_bw = st.number_input("BS's Foals Black-Type Winners", min_value=0, value=int(imputation_medians.get('BS_BW', 5)), step=1); bs_earnings = st.number_input("BS's Foals Earnings ($)", min_value=0, value=int(imputation_medians.get('BS_earnings', 5000000)), step=1000000, format="%d"); bs_aei = st.number_input("BS's Foals AEI", min_value=0.0, max_value=10.0, value=float(imputation_medians.get('BS_aei', 1.10)), step=0.05, format="%.2f")

    submitted = st.form_submit_button("Predict Price", use_container_width=True)


# --- Prediction Logic & Display Area ---
st.divider(); st.markdown("### Prediction Result", unsafe_allow_html=True)
prediction_placeholder = st.empty()

if submitted:
    model_pipeline = st.session_state.model_pipeline; imp_med = st.session_state.imputation_medians
    s_rep_lkp = st.session_state.sire_rep_lookup; s_yr_lkp = st.session_state.sire_year_lookup; s_rep_med = st.session_state.sire_rep_median
    sell_rep_lkp = st.session_state.seller_rep_lookup; sell_rep_med = st.session_state.seller_rep_median
    d_rep_lkp = st.session_state.dam_rep_lookup; d_rep_med = st.session_state.dam_rep_median
    current_sire_name = st.session_state.selected_sire

    if not current_sire_name: st.session_state.prediction_result = ("Error", "Sire Name required.")
    elif not dam_name: st.session_state.prediction_result = ("Error", "Dam Name required.")
    elif not seller_name: st.session_state.prediction_result = ("Error", "Seller/Consignor required.")
    elif model_pipeline is None: st.session_state.prediction_result = ("Error", "Model not loaded.")
    else:
        input_data = { 'Sire': current_sire_name, 'Dam': dam_name, 'PropertyLine1': seller_name, 'Year': CURRENT_YEAR, 'Session': session, 'Color': horse_color_code, 'Sex': sex, 'Brilliant': brill, 'Intermediate': inter, 'Classic': clas, 'Solid': sol, 'Professional': prof, 'Sire_starters': sire_starters, 'Sire_winners': sire_winners, 'Sire_BW': sire_bw, 'Sire_earnings': sire_earnings, 'Sire_aei': sire_aei, 'BS_foals': bs_foals, 'BS_starters': bs_starters, 'BS_winners': bs_winners, 'BS_BW': bs_bw, 'BS_earnings': bs_earnings, 'BS_aei': bs_aei}
        if 'Hip' in FEATURES_TO_USE and 'hip' in locals(): input_data['Hip'] = hip
        # DI/CD are calculated in FE
        for k in imp_med.keys(): input_data.setdefault(k, imp_med.get(k))

        if DEBUG_MODE: st.write("Input data collected:", {k: v for k,v in input_data.items() if pd.notna(v)})

        with st.spinner("Calculating features and predicting price..."):
            try:
                features_df = calculate_features(input_data, FEATURES_TO_USE, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, imp_med, s_rep_lkp, s_yr_lkp, s_rep_med, sell_rep_lkp, sell_rep_med, d_rep_lkp, d_rep_med)
                if DEBUG_MODE: st.dataframe(features_df)

                if 'mapie' not in str(type(model_pipeline)).lower(): st.error("Loaded model not Mapie."); st.session_state.prediction_result = ("Error", "Interval calculation needs Mapie model."); st.rerun()
                else:
                    mapie_alpha = MAPIE_ALPHA
                    y_pred_mapie, y_pis_mapie = model_pipeline.predict(features_df, alpha=mapie_alpha)
                    log_prediction_scalar = float(y_pred_mapie[0]); log_lower_bound = float(y_pis_mapie[0, 0, 0]); log_upper_bound = float(y_pis_mapie[0, 1, 0])
                    predicted_price = np.expm1(log_prediction_scalar); price_lower_bound = max(0, np.expm1(log_lower_bound)); price_upper_bound = np.expm1(log_upper_bound)
                    st.session_state.prediction_result = ("Success", predicted_price, price_lower_bound, price_upper_bound) # Store range

            except Exception as e:
                st.session_state.prediction_result = ("Error", f"An error occurred during prediction: {e}")
                if DEBUG_MODE: st.session_state.prediction_result = ("Error", f"An error occurred: {e}\n{traceback.format_exc()}")
    st.rerun()

# Display result based on session state
if st.session_state.get('prediction_result'):
    result_tuple = st.session_state.prediction_result
    status = result_tuple[0]; value = result_tuple[1]
    with prediction_placeholder.container():
        if status == "Success":
            predicted_price = value; price_lower = result_tuple[2]; price_upper = result_tuple[3]
            # *** Prediction Box includes Plausible Range ***
            st.markdown(f"""
            <div style='background-color: #008000; color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.15); margin: 10px 0; text-align: center;'>
                <h3 style='color: white; margin: 0; font-size: 3.3rem; font-weight: 700;'>${predicted_price:,.0f}</h1>
                <p style='font-size: 1.5rem; margin-top: 10px; margin-bottom: 0; font-weight: 500; opacity: 0.95;'>Plausible Range: ${price_lower:,.0f} - ${price_upper:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            st.info(f"Estimate based on historical data & LightGBM model. Actual prices vary significantly.")
            with st.expander("Technical Details"):
                 st.text(f"Model: LightGBM MapieRegressor ({MODEL_PATH})"); st.text(f"Features Used: {len(FEATURES_TO_USE)}"); st.text(f"Predicted Log Price: {np.log1p(predicted_price):.4f}")
        elif status == "Error":
            st.error(value)
elif st.session_state.get('init_done', False):
     prediction_placeholder.info("Enter yearling details above and click 'Predict Price'.")

# Footer
st.divider(); st.caption("Disclaimer: Statistical model estimates.")