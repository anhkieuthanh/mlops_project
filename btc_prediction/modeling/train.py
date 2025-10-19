from pathlib import Path
import numpy as np

from sklearn.model_selection import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from loguru import logger
from tqdm import tqdm
import typer

from btc_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(data_frame,target_column):
    # Separate features and target
    X_train = data_frame.drop(columns=[target_column])
    y_train = data_frame[target_column]
    
    print("\n--- Đã tách X_train (Features) ---")
    print(X_train)
    print("\n--- Đã tách y_train (Target) ---")
    print(y_train)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    alphas = [0.1, 1.0, 10.0]
    model = RidgeCV(alphas=alphas, store_cv_values=True)
    model.fit(X_train, y_train)

    # Logger Info
    logger.info(f"\nModel đã được huấn luyện xong!")
    logger.info(f"Alpha tối ưu được tìm thấy: {model.alpha_}")
    logger.info(f"Hệ số chặn (intercept): {model.intercept_}")
    logger.info(f"Các hệ số hồi quy (coefficients): \n{model.coef_}")

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Logger Info
    logger.info(f"\nĐánh giá mô hình trên tập huấn luyện:")
    logger.info(f"MSE: {mse}")
    logger.info(f"RMSE: {rmse}")
    logger.info(f"R²: {r2}")

    return model

if __name__ == "__main__":
    app()
