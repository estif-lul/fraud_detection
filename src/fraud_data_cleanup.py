from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import pandas as pd

def clean_fraud_data(fraud_df):

    for col in fraud_df.columns:
        print(f"{col}: {fraud_df[col].dtype}, Nulls: {fraud_df[col].isnull().sum()}, Unique: {fraud_df[col].nunique()}")

    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'], errors='coerce')
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'], errors='coerce')

    # Drop rows where conversion failed (optional: log how many)
    fraud_df = fraud_df.dropna(subset=['purchase_time', 'signup_time'])
    
    invalid_timestamps = fraud_df[fraud_df['purchase_time'] < fraud_df['signup_time']]
    print(f"Invalid timestamps: {len(invalid_timestamps)}")
    # Option: Drop or correct if known registration lag issues
    fraud_df = fraud_df[fraud_df['purchase_time'] >= fraud_df['signup_time']]

    device_freq = fraud_df['device_id'].value_counts()
    high_freq_devices = device_freq[device_freq > 100].index
    fraud_df['is_high_freq_device'] = fraud_df['device_id'].isin(high_freq_devices).astype(int)

    fraud_df['ip_freq'] = fraud_df.groupby('ip_address')['ip_address'].transform('count')

    fraud_df['sex'] = fraud_df['sex'].str.upper().replace({'UNKNOWN': None})
    fraud_df['browser'] = fraud_df['browser'].str.strip().str.title()
    fraud_df['source'] = fraud_df['source'].str.strip().str.title()

    mixed_type_cols = [col for col in fraud_df.columns if fraud_df[col].apply(type).nunique() > 1]
    print("Mixed type columns:", mixed_type_cols)

    Q1 = fraud_df['purchase_value'].quantile(0.25)
    Q3 = fraud_df['purchase_value'].quantile(0.75)
    IQR = Q3 - Q1

    outlier_mask = (fraud_df['purchase_value'] < Q1 - 1.5*IQR) | (fraud_df['purchase_value'] > Q3 + 1.5*IQR)
    fraud_df['is_outlier_purchase'] = outlier_mask.astype(int)


    fraud_df['age'] = fraud_df['age'].fillna(fraud_df.groupby('sex')['age'].transform('median'))
    fraud_df['browser'] = fraud_df['browser'].fillna("Unknown")

    return fraud_df.copy()


def sanity_report(df):
    print("Shape:", df.shape)
    print("Any nulls?", df.isnull().any().any())
    print("Dtypes:", df.dtypes.value_counts())