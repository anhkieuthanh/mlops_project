from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from btc_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(trained_model_, input_data_):
    
    predicted_price = trained_model_.predict(input_data_)
    logger.info(f"\nDự đoán giá BTC hoàn tất!")
    logger.info(f"Giá BTC dự đoán: {predicted_price[0]}")
    
    return predicted_price

if __name__ == "__main__":
    app()
