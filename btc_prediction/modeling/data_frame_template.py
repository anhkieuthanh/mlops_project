import pandas as pd

data_frame_template ={
  "Open": [0.0],
  "High": [0.0],
  "Low": [0.0],
  "Close": [0.0],
  "Volume": [0.0],
  "Quote_asset_volume": [0.0],
  "Number_of_trades": [0],
  "Taker_buy_base_asset_volume": [0.0],
  "Taker_buy_base_quote_volume": [0.0],
  "VWAP": [0.0],
  "RSI": [0.0],
  "DCdown": [0.0],
  "DCup": [0.0],
  "DCmid": [0.0],
  "BollingerBasis": [0.0],
  "BollingerUpper": [0.0],
  "BollingerLower": [0.0],
}

data_frame_template = pd.DataFrame(data_frame_template)