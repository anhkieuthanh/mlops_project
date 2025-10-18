import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, timedelta
from minio import Minio

# Import functions từ các file khác
from extract import (
    get_hour, get_day, get_month, get_year,
    save_to_minio_partitioned, 
    get_data as extract_binance_data
)

from transform import (
    read_raw_data_from_minio,
    save_processed_data_to_minio,
    generate_features
)

# =================== HELPER FUNCTIONS ===================
def save_to_minio_with_type(df, minio_client, bucket_name, data_type="raw"):
    """
    Wrapper function để lưu với data_type khác nhau
    """
    if data_type == "raw":
        save_to_minio_partitioned(df, minio_client, bucket_name)
    else:
        save_processed_data_to_minio(df, minio_client, bucket_name)

# =================== MAIN PIPELINE FUNCTION ===================
def run_pipeline(start_date=None, end_date=None):
    """
    Chạy pipeline Bitcoin data: Extract từ Binance API và Transform tạo features
    
    Args:
        start_date: Start date cho extract data
    """
    # Các tham số cố định
    mode = 'full_pipeline'
    minio_host = 'localhost:9000'
    minio_access_key = 'admin'
    minio_secret_key = '12345678'
    bucket_name = 'btc-prediction'
    api_key = "RVMmqTn1fS3qfq35f4HA2z93T0NCIGHsrqqP9lvbb639Aor9OLw1h5B2hM6jWffq"
    api_secret = "vTaHbz55Gn0z40Gwqt8ZCmDmWHnAWRNRHL6QjS0xpO6kTfDhgOSP9wvRU96wEM4g"
    
    # Setup MinIO client
    minio_client = Minio(
        minio_host,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )
    
    if mode in ['extract_only', 'full_pipeline']:
        print("=== BƯỚC 1: EXTRACT DỮ LIỆU ===")
        
        # Extract từ Binance sử dụng function từ extract.py
        raw_data = extract_binance_data(
            api_key=api_key,
            api_secret=api_secret,
            minio_client=minio_client,
            bucket_name=bucket_name,
            start_date=start_date,
            end_date=end_date
        )
                
        if mode == 'extract_only':
            return raw_data
    
    if mode in ['transform_only', 'full_pipeline']:
        print("=== BƯỚC 2: TRANSFORM DỮ LIỆU ===")
        
        # Nếu chạy full pipeline, dùng raw_data từ bước trước
        # Nếu chạy transform_only, đọc từ MinIO
        if mode == 'transform_only':
            raw_data = read_raw_data_from_minio(
                bucket_name=bucket_name,
                start_date=start_date
            )
            
            if raw_data is None:
                print("Không tìm thấy raw data. Hãy chạy extract trước.")
                return None
        
        # Nếu chạy full pipeline, đọc lại để có dữ liệu từ MinIO  
        if mode == 'full_pipeline':
            raw_data_from_minio = read_raw_data_from_minio(
                bucket_name=bucket_name,
                start_date=start_date
            )
            if raw_data_from_minio is not None:
                raw_data = raw_data_from_minio
        
        # Transform data sử dụng function từ transform.py
        processed_data = generate_features(dataframes=raw_data.copy())
        
        # Lưu processed data vào MinIO
        save_processed_data_to_minio(
            df=processed_data,
            minio_client=minio_client,
            bucket_name=bucket_name
        )
        
        return processed_data
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bitcoin Data Pipeline - Extract và Transform')
    parser.add_argument('--start-date', type=str, help='Start date (e.g., "1 Jan, 2020" hoặc "2025-01-01")')
    args = parser.parse_args()

    # convert to datetime
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.now()

    start_timestamp_ms = int(start_date.timestamp() * 1000)
    end_timestamp_ms = int(end_date.timestamp() * 1000)

    # Chạy pipeline
    result = run_pipeline(start_date=start_timestamp_ms, end_date=end_timestamp_ms)
    if result is not None:
        print(f"Thời gian từ: {result['datetime'].min()}")
        print(f"Thời gian đến: {result['datetime'].max()}")
            