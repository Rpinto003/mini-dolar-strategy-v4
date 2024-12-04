import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

class MarketDataLoader:
    def __init__(self, db_path: str = None):
        """Initialize the market data loader.
        
        Args:
            db_path: Path to the .db file. If None, will use default path.
        """
        self.logger = logging.getLogger(__name__)
        
        if db_path is None:
            # Use absolute path based on current directory
            current_dir = os.getcwd()
            db_path = os.path.join(current_dir, 'src', 'data', 'database', 'candles.db')
        else:
            # If path provided, convert to absolute
            db_path = os.path.abspath(db_path)
            
        self.db_path = db_path
        self.logger.info(f"Using database: {self.db_path}")
        
        # Check if file exists
        if not os.path.exists(self.db_path):
            self.logger.error(f"Database file not found: {self.db_path}")
    
    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load data from SQLite database.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Attempting to load data from {self.db_path}")
            
            query = "SELECT * FROM candles"
            conditions = []
            
            if start_date:
                conditions.append(f"time >= '{start_date}'")
            if end_date:
                conditions.append(f"time <= '{end_date}'")
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY time ASC"
            
            self.logger.info(f"Executing query: {query}")
            
            # Connect to database and read data
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
            self.logger.info(f"Data loaded: {len(df)} records")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    def get_minute_data(self, 
                       interval: int = 5,
                       start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
        """Return data in specific minute intervals."""
        df = self.load_data(start_date, end_date)
        if df.empty:
            return df
            
        # Resampling to desired interval
        rule = f"{interval}T"
        resampled_data = pd.DataFrame()
        resampled_data['open'] = df['open'].resample(rule).first()
        resampled_data['high'] = df['high'].resample(rule).max()
        resampled_data['low'] = df['low'].resample(rule).min()
        resampled_data['close'] = df['close'].resample(rule).last()
        resampled_data['volume'] = df['real_volume'].resample(rule).sum()
        
        return resampled_data.dropna()
        
    def get_latest_data(self, 
                       lookback: int = 100) -> pd.DataFrame:
        """Get most recent market data."""
        try:
            query = f"""
            SELECT *
            FROM candles 
            ORDER BY time DESC
            LIMIT {lookback}
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading latest data: {str(e)}")
            return pd.DataFrame()