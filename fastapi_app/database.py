"""
Database setup and models for the FastAPI trading application
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

class TradingDatabase:
    def __init__(self, db_path: str = "trading.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    filled_quantity REAL DEFAULT 0,
                    filled_price REAL
                )
            """)

            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    quantity REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    last_updated DATETIME NOT NULL
                )
            """)

            # Strategies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    strategy_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    symbols TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    last_updated DATETIME NOT NULL
                )
            """)

            # Backtests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT UNIQUE NOT NULL,
                    strategy_name TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    initial_capital REAL NOT NULL,
                    symbols TEXT NOT NULL,
                    status TEXT NOT NULL,
                    results TEXT,
                    created_at DATETIME NOT NULL,
                    completed_at DATETIME
                )
            """)

            # Portfolio snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    daily_pnl REAL,
                    total_pnl REAL,
                    timestamp DATETIME NOT NULL
                )
            """)

            # Market data cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    change_amount REAL,
                    change_percent REAL,
                    volume INTEGER,
                    timestamp DATETIME NOT NULL,
                    UNIQUE(symbol, timestamp)
                )
            """)

            conn.commit()
            print("✅ Database initialized successfully")

    def save_order(self, order_data: Dict[str, Any]):
        """Save order to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO orders
                (order_id, symbol, side, order_type, quantity, price, status, timestamp, filled_quantity, filled_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_data.get('order_id'),
                order_data.get('symbol'),
                order_data.get('side'),
                order_data.get('order_type'),
                order_data.get('quantity'),
                order_data.get('price'),
                order_data.get('status'),
                order_data.get('timestamp', datetime.now()),
                order_data.get('filled_quantity', 0),
                order_data.get('filled_price')
            ))
            conn.commit()

    def get_orders(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get orders from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM orders
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def save_position(self, symbol: str, quantity: float, avg_cost: float):
        """Save position to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if quantity == 0:
                # Remove position if quantity is 0
                cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO positions
                    (symbol, quantity, avg_cost, last_updated)
                    VALUES (?, ?, ?, ?)
                """, (symbol, quantity, avg_cost, datetime.now()))
            conn.commit()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE quantity != 0")
            return [dict(row) for row in cursor.fetchall()]

    def save_strategy(self, strategy_data: Dict[str, Any]):
        """Save strategy to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO strategies
                (name, strategy_type, parameters, symbols, status, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_data.get('name'),
                strategy_data.get('strategy_type'),
                json.dumps(strategy_data.get('parameters', {})),
                json.dumps(strategy_data.get('symbols', [])),
                strategy_data.get('status', 'inactive'),
                strategy_data.get('created_at', datetime.now()),
                datetime.now()
            ))
            conn.commit()

    def get_strategies(self) -> List[Dict[str, Any]]:
        """Get strategies from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM strategies ORDER BY created_at DESC")
            strategies = []
            for row in cursor.fetchall():
                strategy = dict(row)
                strategy['parameters'] = json.loads(strategy['parameters'])
                strategy['symbols'] = json.loads(strategy['symbols'])
                strategies.append(strategy)
            return strategies

    def save_backtest(self, backtest_data: Dict[str, Any]):
        """Save backtest to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO backtests
                (backtest_id, strategy_name, start_date, end_date, initial_capital, symbols, status, results, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                backtest_data.get('backtest_id'),
                backtest_data.get('strategy_name'),
                backtest_data.get('start_date'),
                backtest_data.get('end_date'),
                backtest_data.get('initial_capital'),
                json.dumps(backtest_data.get('symbols', [])),
                backtest_data.get('status'),
                json.dumps(backtest_data.get('results')) if backtest_data.get('results') else None,
                backtest_data.get('created_at', datetime.now()),
                backtest_data.get('completed_at')
            ))
            conn.commit()

    def get_backtests(self) -> List[Dict[str, Any]]:
        """Get backtests from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM backtests ORDER BY created_at DESC")
            backtests = []
            for row in cursor.fetchall():
                backtest = dict(row)
                backtest['symbols'] = json.loads(backtest['symbols'])
                if backtest['results']:
                    backtest['results'] = json.loads(backtest['results'])
                backtests.append(backtest)
            return backtests

    def save_portfolio_snapshot(self, snapshot: Dict[str, Any]):
        """Save portfolio snapshot to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO portfolio_snapshots
                (total_value, cash, positions_value, daily_pnl, total_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                snapshot.get('total_value'),
                snapshot.get('cash'),
                snapshot.get('positions_value'),
                snapshot.get('daily_pnl'),
                snapshot.get('total_pnl'),
                snapshot.get('timestamp', datetime.now())
            ))
            conn.commit()

    def get_portfolio_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get portfolio history from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM portfolio_snapshots
                WHERE timestamp > datetime('now', '-{} days')
                ORDER BY timestamp DESC
            """.format(days))
            return [dict(row) for row in cursor.fetchall()]

    def save_market_data(self, symbol: str, price: float, change_amount: float = None,
                        change_percent: float = None, volume: int = None):
        """Save market data to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO market_data
                (symbol, price, change_amount, change_percent, volume, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, price, change_amount, change_percent, volume, datetime.now()))
            conn.commit()

    def get_market_data(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get market data from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if symbol:
                cursor.execute("""
                    SELECT * FROM market_data
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (symbol,))
            else:
                cursor.execute("""
                    SELECT DISTINCT symbol,
                           FIRST_VALUE(price) OVER (PARTITION BY symbol ORDER BY timestamp DESC) as price,
                           FIRST_VALUE(change_amount) OVER (PARTITION BY symbol ORDER BY timestamp DESC) as change_amount,
                           FIRST_VALUE(change_percent) OVER (PARTITION BY symbol ORDER BY timestamp DESC) as change_percent,
                           FIRST_VALUE(volume) OVER (PARTITION BY symbol ORDER BY timestamp DESC) as volume,
                           FIRST_VALUE(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp DESC) as timestamp
                    FROM market_data
                    ORDER BY timestamp DESC
                """)
            return [dict(row) for row in cursor.fetchall()]

# Global database instance
db = TradingDatabase()