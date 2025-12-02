"""
Firebase Firestore Database module for TraderQ
Handles all database operations for user data, replacing JSON file storage
"""

from firebase_admin import firestore
from typing import Dict, List, Optional
from datetime import datetime
import json


class FirestoreDB:
    """
    Firestore database wrapper for TraderQ
    Provides methods to store and retrieve user data
    """

    def __init__(self):
        """Initialize Firestore client"""
        self.db = firestore.client()

    # ==================== Custom Tickers ====================

    def get_custom_tickers(self, user_id: str) -> Dict:
        """
        Get user's custom tickers

        Returns:
            Dict with 'custom' and 'selected' keys containing Stocks and Crypto arrays
        """
        try:
            # Get stocks
            stocks_ref = self.db.collection('users').document(user_id)\
                .collection('customTickers').document('stocks')
            stocks_doc = stocks_ref.get()

            # Get crypto
            crypto_ref = self.db.collection('users').document(user_id)\
                .collection('customTickers').document('crypto')
            crypto_doc = crypto_ref.get()

            # Get selected tickers
            selected = self.get_selected_tickers(user_id)

            return {
                "custom": {
                    "Stocks": stocks_doc.to_dict().get('tickers', []) if stocks_doc.exists else [],
                    "Crypto": crypto_doc.to_dict().get('tickers', []) if crypto_doc.exists else []
                },
                "selected": selected
            }
        except Exception as e:
            print(f"Error getting custom tickers: {e}")
            return {"custom": {"Stocks": [], "Crypto": []}, "selected": {"Stocks": [], "Crypto": []}}

    def save_custom_tickers(self, user_id: str, data: Dict):
        """
        Save user's custom tickers

        Args:
            user_id: User's Firebase UID
            data: Dict with 'custom' key containing {'Stocks': [...], 'Crypto': [...]}
        """
        try:
            batch = self.db.batch()

            # Save stocks
            stocks_ref = self.db.collection('users').document(user_id)\
                .collection('customTickers').document('stocks')
            batch.set(stocks_ref, {
                'tickers': data.get('custom', {}).get('Stocks', []),
                'updatedAt': firestore.SERVER_TIMESTAMP
            })

            # Save crypto
            crypto_ref = self.db.collection('users').document(user_id)\
                .collection('customTickers').document('crypto')
            batch.set(crypto_ref, {
                'tickers': data.get('custom', {}).get('Crypto', []),
                'updatedAt': firestore.SERVER_TIMESTAMP
            })

            batch.commit()
        except Exception as e:
            print(f"Error saving custom tickers: {e}")

    def get_selected_tickers(self, user_id: str) -> Dict:
        """Get user's selected tickers"""
        try:
            stocks_ref = self.db.collection('users').document(user_id)\
                .collection('selectedTickers').document('stocks')
            stocks_doc = stocks_ref.get()

            crypto_ref = self.db.collection('users').document(user_id)\
                .collection('selectedTickers').document('crypto')
            crypto_doc = crypto_ref.get()

            return {
                "Stocks": stocks_doc.to_dict().get('tickers', []) if stocks_doc.exists else [],
                "Crypto": crypto_doc.to_dict().get('tickers', []) if crypto_doc.exists else []
            }
        except Exception as e:
            print(f"Error getting selected tickers: {e}")
            return {"Stocks": [], "Crypto": []}

    def save_selected_tickers(self, user_id: str, selected: Dict):
        """Save user's selected tickers"""
        try:
            batch = self.db.batch()

            stocks_ref = self.db.collection('users').document(user_id)\
                .collection('selectedTickers').document('stocks')
            batch.set(stocks_ref, {
                'tickers': selected.get('Stocks', []),
                'updatedAt': firestore.SERVER_TIMESTAMP
            })

            crypto_ref = self.db.collection('users').document(user_id)\
                .collection('selectedTickers').document('crypto')
            batch.set(crypto_ref, {
                'tickers': selected.get('Crypto', []),
                'updatedAt': firestore.SERVER_TIMESTAMP
            })

            batch.commit()
        except Exception as e:
            print(f"Error saving selected tickers: {e}")

    # ==================== Alerts ====================

    def get_alerts(self, user_id: str) -> List[Dict]:
        """Get all alerts for user"""
        try:
            alerts_ref = self.db.collection('users').document(user_id).collection('alerts')
            alerts = alerts_ref.stream()

            result = []
            for alert in alerts:
                alert_data = alert.to_dict()
                alert_data['id'] = alert.id
                # Convert timestamp to ISO string
                if 'created' in alert_data and hasattr(alert_data['created'], 'isoformat'):
                    alert_data['created'] = alert_data['created'].isoformat()
                result.append(alert_data)

            return result
        except Exception as e:
            print(f"Error getting alerts: {e}")
            return []

    def save_alerts(self, user_id: str, alerts: List[Dict]):
        """
        Save alerts for user (replaces all existing alerts)

        Args:
            user_id: User's Firebase UID
            alerts: List of alert dictionaries
        """
        try:
            # Delete all existing alerts
            alerts_ref = self.db.collection('users').document(user_id).collection('alerts')
            existing_alerts = alerts_ref.stream()
            batch = self.db.batch()

            for alert in existing_alerts:
                batch.delete(alert.reference)

            # Add new alerts
            for alert in alerts:
                alert_data = alert.copy()
                # Remove ID if present (Firestore generates its own)
                alert_data.pop('id', None)
                # Convert created timestamp if it's a string
                if 'created' in alert_data and isinstance(alert_data['created'], str):
                    try:
                        alert_data['created'] = datetime.fromisoformat(alert_data['created'])
                    except:
                        alert_data['created'] = firestore.SERVER_TIMESTAMP
                elif 'created' not in alert_data:
                    alert_data['created'] = firestore.SERVER_TIMESTAMP

                doc_ref = alerts_ref.document()
                batch.set(doc_ref, alert_data)

            batch.commit()
        except Exception as e:
            print(f"Error saving alerts: {e}")

    def create_alert(self, user_id: str, alert: Dict) -> str:
        """
        Create a single alert

        Returns:
            Alert ID
        """
        try:
            alert_data = alert.copy()
            if 'created' not in alert_data:
                alert_data['created'] = firestore.SERVER_TIMESTAMP

            doc_ref = self.db.collection('users').document(user_id)\
                .collection('alerts').document()
            doc_ref.set(alert_data)
            return doc_ref.id
        except Exception as e:
            print(f"Error creating alert: {e}")
            return ""

    def delete_alert(self, user_id: str, alert_id: str):
        """Delete a specific alert"""
        try:
            self.db.collection('users').document(user_id)\
                .collection('alerts').document(alert_id).delete()
        except Exception as e:
            print(f"Error deleting alert: {e}")

    # ==================== Alert History ====================

    def get_alert_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get alert history for user"""
        try:
            history_ref = self.db.collection('users').document(user_id)\
                .collection('alertHistory').order_by('triggeredAt', direction=firestore.Query.DESCENDING)\
                .limit(limit)

            history = history_ref.stream()
            result = []
            for item in history:
                item_data = item.to_dict()
                item_data['id'] = item.id
                # Convert timestamp
                if 'triggeredAt' in item_data and hasattr(item_data['triggeredAt'], 'isoformat'):
                    item_data['triggeredAt'] = item_data['triggeredAt'].isoformat()
                result.append(item_data)

            return result
        except Exception as e:
            print(f"Error getting alert history: {e}")
            return []

    def add_alert_history(self, user_id: str, history_item: Dict):
        """Add an item to alert history"""
        try:
            history_data = history_item.copy()
            if 'triggeredAt' not in history_data:
                history_data['triggeredAt'] = firestore.SERVER_TIMESTAMP

            self.db.collection('users').document(user_id)\
                .collection('alertHistory').add(history_data)
        except Exception as e:
            print(f"Error adding alert history: {e}")

    # ==================== Portfolio ====================

    def get_portfolio(self, user_id: str) -> Dict:
        """Get user's portfolio"""
        try:
            portfolio_ref = self.db.collection('users').document(user_id)\
                .collection('portfolio').document('current')
            portfolio_doc = portfolio_ref.get()

            if portfolio_doc.exists:
                return portfolio_doc.to_dict()
            else:
                return {"tickers": [], "weights": {}}
        except Exception as e:
            print(f"Error getting portfolio: {e}")
            return {"tickers": [], "weights": {}}

    def save_portfolio(self, user_id: str, portfolio: Dict):
        """Save user's portfolio"""
        try:
            portfolio_ref = self.db.collection('users').document(user_id)\
                .collection('portfolio').document('current')

            portfolio_ref.set({
                'tickers': portfolio.get('tickers', []),
                'weights': portfolio.get('weights', {}),
                'updatedAt': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            print(f"Error saving portfolio: {e}")

    # ==================== Watchlists ====================

    def get_watchlists(self, user_id: str) -> Dict:
        """Get all watchlists for user"""
        try:
            watchlists_ref = self.db.collection('users').document(user_id)\
                .collection('watchlists')
            watchlists = watchlists_ref.stream()

            result = {}
            for watchlist in watchlists:
                watchlist_data = watchlist.to_dict()
                result[watchlist.id] = watchlist_data

            return result
        except Exception as e:
            print(f"Error getting watchlists: {e}")
            return {}

    def save_watchlists(self, user_id: str, watchlists: Dict):
        """Save all watchlists for user"""
        try:
            # Delete all existing watchlists
            watchlists_ref = self.db.collection('users').document(user_id)\
                .collection('watchlists')
            existing = watchlists_ref.stream()
            batch = self.db.batch()

            for wl in existing:
                batch.delete(wl.reference)

            # Add new watchlists
            for name, data in watchlists.items():
                doc_ref = watchlists_ref.document(name)
                watchlist_data = {
                    'tickers': data.get('tickers', []),
                    'mode': data.get('mode', 'Stocks'),
                    'created': data.get('created', firestore.SERVER_TIMESTAMP),
                    'updatedAt': firestore.SERVER_TIMESTAMP
                }
                batch.set(doc_ref, watchlist_data)

            batch.commit()
        except Exception as e:
            print(f"Error saving watchlists: {e}")

    def create_watchlist(self, user_id: str, name: str, tickers: List[str], mode: str = "Stocks"):
        """Create a single watchlist"""
        try:
            self.db.collection('users').document(user_id)\
                .collection('watchlists').document(name).set({
                    'tickers': tickers,
                    'mode': mode,
                    'created': firestore.SERVER_TIMESTAMP,
                    'updatedAt': firestore.SERVER_TIMESTAMP
                })
        except Exception as e:
            print(f"Error creating watchlist: {e}")

    def delete_watchlist(self, user_id: str, name: str):
        """Delete a watchlist"""
        try:
            self.db.collection('users').document(user_id)\
                .collection('watchlists').document(name).delete()
        except Exception as e:
            print(f"Error deleting watchlist: {e}")

    # ==================== Trade Journal ====================

    def get_trade_journal(self, user_id: str) -> List[Dict]:
        """Get all trades from trade journal"""
        try:
            journal_ref = self.db.collection('users').document(user_id)\
                .collection('tradeJournal').order_by('date', direction=firestore.Query.DESCENDING)
            trades = journal_ref.stream()

            result = []
            for trade in trades:
                trade_data = trade.to_dict()
                trade_data['id'] = trade.id
                # Convert timestamp
                if 'date' in trade_data and hasattr(trade_data['date'], 'isoformat'):
                    trade_data['date'] = trade_data['date'].isoformat()
                result.append(trade_data)

            return result
        except Exception as e:
            print(f"Error getting trade journal: {e}")
            return []

    def save_trade_journal(self, user_id: str, trades: List[Dict]):
        """Save trade journal (replaces all existing trades)"""
        try:
            # Delete all existing trades
            journal_ref = self.db.collection('users').document(user_id)\
                .collection('tradeJournal')
            existing_trades = journal_ref.stream()
            batch = self.db.batch()

            for trade in existing_trades:
                batch.delete(trade.reference)

            # Add new trades
            for trade in trades:
                trade_data = trade.copy()
                trade_data.pop('id', None)

                # Convert date if string
                if 'date' in trade_data and isinstance(trade_data['date'], str):
                    try:
                        trade_data['date'] = datetime.fromisoformat(trade_data['date'])
                    except:
                        pass

                doc_ref = journal_ref.document()
                batch.set(doc_ref, trade_data)

            batch.commit()
        except Exception as e:
            print(f"Error saving trade journal: {e}")

    def add_trade(self, user_id: str, trade: Dict) -> str:
        """
        Add a single trade to journal

        Returns:
            Trade ID
        """
        try:
            trade_data = trade.copy()

            # Convert date if string
            if 'date' in trade_data and isinstance(trade_data['date'], str):
                try:
                    trade_data['date'] = datetime.fromisoformat(trade_data['date'])
                except:
                    trade_data['date'] = firestore.SERVER_TIMESTAMP

            doc_ref = self.db.collection('users').document(user_id)\
                .collection('tradeJournal').document()
            doc_ref.set(trade_data)
            return doc_ref.id
        except Exception as e:
            print(f"Error adding trade: {e}")
            return ""

    # ==================== Cross History ====================

    def get_cross_history(self, user_id: str) -> Dict:
        """Get cross history for all tickers"""
        try:
            history_ref = self.db.collection('users').document(user_id)\
                .collection('crossHistory')
            tickers = history_ref.stream()

            result = {}
            for ticker_doc in tickers:
                ticker = ticker_doc.id
                ticker_data = ticker_doc.to_dict()

                # Get events if stored as array in document
                if 'events' in ticker_data:
                    result[ticker] = ticker_data['events']

            return result
        except Exception as e:
            print(f"Error getting cross history: {e}")
            return {}

    def save_cross_history(self, user_id: str, history: Dict):
        """Save cross history for all tickers"""
        try:
            batch = self.db.batch()
            history_ref = self.db.collection('users').document(user_id)\
                .collection('crossHistory')

            for ticker, events in history.items():
                doc_ref = history_ref.document(ticker)
                batch.set(doc_ref, {
                    'events': events,
                    'updatedAt': firestore.SERVER_TIMESTAMP
                })

            batch.commit()
        except Exception as e:
            print(f"Error saving cross history: {e}")

    # ==================== User Preferences ====================

    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        try:
            user_ref = self.db.collection('users').document(user_id)
            user_doc = user_ref.get()

            if user_doc.exists:
                return user_doc.to_dict().get('preferences', {})
            else:
                return {}
        except Exception as e:
            print(f"Error getting user preferences: {e}")
            return {}

    def save_user_preferences(self, user_id: str, preferences: Dict):
        """Save user preferences"""
        try:
            self.db.collection('users').document(user_id).update({
                'preferences': preferences
            })
        except Exception as e:
            print(f"Error saving user preferences: {e}")

    # ==================== Email & Kraken Config (Temporary - will move to Secret Manager) ====================

    def get_email_config(self, user_id: str) -> Dict:
        """
        Get email configuration (temporary storage in Firestore)
        TODO: Move to Secret Manager in Phase 2
        """
        try:
            config_ref = self.db.collection('users').document(user_id)\
                .collection('config').document('email')
            config_doc = config_ref.get()

            if config_doc.exists:
                return config_doc.to_dict()
            else:
                return {}
        except Exception as e:
            print(f"Error getting email config: {e}")
            return {}

    def save_email_config(self, user_id: str, config: Dict):
        """
        Save email configuration (temporary storage in Firestore)
        TODO: Move to Secret Manager in Phase 2
        """
        try:
            self.db.collection('users').document(user_id)\
                .collection('config').document('email').set(config)
        except Exception as e:
            print(f"Error saving email config: {e}")

    def get_kraken_config(self, user_id: str) -> Dict:
        """
        Get Kraken API configuration (temporary storage in Firestore)
        TODO: Move to Secret Manager in Phase 2
        """
        try:
            config_ref = self.db.collection('users').document(user_id)\
                .collection('config').document('kraken')
            config_doc = config_ref.get()

            if config_doc.exists:
                return config_doc.to_dict()
            else:
                return {}
        except Exception as e:
            print(f"Error getting Kraken config: {e}")
            return {}

    def save_kraken_config(self, user_id: str, config: Dict):
        """
        Save Kraken API configuration (temporary storage in Firestore)
        TODO: Move to Secret Manager in Phase 2
        """
        try:
            self.db.collection('users').document(user_id)\
                .collection('config').document('kraken').set(config)
        except Exception as e:
            print(f"Error saving Kraken config: {e}")
