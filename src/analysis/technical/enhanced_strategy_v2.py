import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import talib

class EnhancedTechnicalStrategyV2:
    def __init__(self,
                 rsi_period=14,
                 ma_fast=9,
                 ma_slow=21,
                 volume_profile_periods=20,
                 gap_threshold=0.2,
                 session_times=None):
        
        self.rsi_period = rsi_period
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.volume_profile_periods = volume_profile_periods
        self.gap_threshold = gap_threshold
        self.session_times = session_times or {
            'morning_start': '09:00',
            'morning_end': '11:00',
            'afternoon_start': '14:00',
            'afternoon_end': '16:00'
        }
        
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_split=50,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def calculate_volume_profile(self, data):
        df = data.copy()
        price_bins = pd.qcut(df['close'], q=10, labels=False)
        volume_profile = df.groupby(price_bins)['volume'].sum()
        high_volume_zones = volume_profile[volume_profile > volume_profile.mean()]
        return high_volume_zones.index
        
    def detect_gaps(self, data):
        df = data.copy()
        df['prev_close'] = df['close'].shift(1)
        df['gap_size'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
        df['gap_type'] = np.where(
            abs(df['gap_size']) > self.gap_threshold,
            np.where(df['gap_size'] > 0, 'up_gap', 'down_gap'),
            'no_gap'
        )
        return df
        
    def session_filter(self, data):
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        morning_mask = df.index.strftime('%H:%M').between(
            self.session_times['morning_start'],
            self.session_times['morning_end']
        )
        afternoon_mask = df.index.strftime('%H:%M').between(
            self.session_times['afternoon_start'],
            self.session_times['afternoon_end']
        )
        df['session_active'] = morning_mask | afternoon_mask
        return df
        
    def calculate_orderflow_indicators(self, data):
        df = data.copy()
        df['delta'] = np.where(df['close'] > df['open'],
                             df['volume'],
                             -df['volume'])
        df['buying_pressure'] = df['high'] - df['close']
        df['selling_pressure'] = df['close'] - df['low']
        df['pressure_ratio'] = df['buying_pressure'] / df['selling_pressure']
        df['cum_delta'] = df['delta'].cumsum()
        df['cum_volume'] = df['volume'].cumsum()
        return df
        
    def prepare_features(self, data):
        df = data.copy()
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        df['ma_fast'] = talib.EMA(df['close'], timeperiod=self.ma_fast)
        df['ma_slow'] = talib.EMA(df['close'], timeperiod=self.ma_slow)
        df = self.calculate_orderflow_indicators(df)
        df = self.detect_gaps(df)
        df = self.session_filter(df)
        feature_cols = [
            'rsi', 'pressure_ratio', 'cum_delta', 'gap_size',
            'buying_pressure', 'selling_pressure'
        ]
        X = df[feature_cols].copy()
        X = self.scaler.fit_transform(X)
        return X, df
        
    def train_model(self, data, future_periods=5):
        X, df = self.prepare_features(data)
        df['future_return'] = df['close'].pct_change(future_periods).shift(-future_periods)
        df['label'] = np.where(df['future_return'] > 0, 1, 0)
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = df['label'].iloc[train_idx]
            self.model.fit(X_train, y_train)
            
    def generate_signals(self, data):
        X, df = self.prepare_features(data)
        df['ml_prob'] = self.model.predict_proba(X)[:, 1]
        df['signal'] = 0
        long_conditions = (
            (df['ml_prob'] > 0.7) &
            (df['session_active']) &
            (df['pressure_ratio'] > 1.2) &
            (df['gap_type'] != 'up_gap')
        )
        short_conditions = (
            (df['ml_prob'] < 0.3) &
            (df['session_active']) &
            (df['pressure_ratio'] < 0.8) &
            (df['gap_type'] != 'down_gap')
        )
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        return df

    def add_risk_management(self, data):
        df = data.copy()
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['stop_loss'] = np.where(
            df['signal'] == 1,
            df['close'] - df['atr'] * 2,
            df['close'] + df['atr'] * 2
        )
        df['take_profit'] = np.where(
            df['signal'] == 1,
            df['close'] + df['atr'] * 3,
            df['close'] - df['atr'] * 3
        )
        df['breakeven_level'] = np.where(
            df['signal'] == 1,
            df['close'] + df['atr'],
            df['close'] - df['atr']
        )
        return df