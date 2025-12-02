import unittest
from unittest.mock import MagicMock

# --- Functions to Test (Copied from app.py to avoid Streamlit dependencies during test) ---

def validate_bracket_prices(side: str, entry_price: float, stop_loss_price: float, take_profit_price: float) -> dict:
    """
    Validate bracket order price levels make sense.
    """
    if not entry_price or entry_price <= 0:
        return {"valid": False, "error": "Entry price must be greater than 0"}
    
    if not stop_loss_price or stop_loss_price <= 0:
        return {"valid": False, "error": "Stop-loss price must be greater than 0"}
    
    if not take_profit_price or take_profit_price <= 0:
        return {"valid": False, "error": "Take-profit price must be greater than 0"}
    
    side_lower = side.lower()
    
    if side_lower == "buy":
        # For buy orders: SL below entry, TP above entry
        if stop_loss_price >= entry_price:
            return {"valid": False, "error": "For BUY orders, stop-loss must be BELOW entry price"}
        if take_profit_price <= entry_price:
            return {"valid": False, "error": "For BUY orders, take-profit must be ABOVE entry price"}
    elif side_lower == "sell":
        # For sell orders: SL above entry, TP below entry
        if stop_loss_price <= entry_price:
            return {"valid": False, "error": "For SELL orders, stop-loss must be ABOVE entry price"}
        if take_profit_price >= entry_price:
            return {"valid": False, "error": "For SELL orders, take-profit must be BELOW entry price"}
    else:
        return {"valid": False, "error": "Side must be 'buy' or 'sell'"}
    
    return {"valid": True}

def calculate_bracket_metrics(entry_price: float, stop_loss_price: float, take_profit_price: float, volume: float) -> dict:
    """
    Calculate risk/reward metrics for bracket order.
    """
    risk_per_unit = abs(entry_price - stop_loss_price)
    reward_per_unit = abs(take_profit_price - entry_price)
    
    total_risk = risk_per_unit * volume
    total_reward = reward_per_unit * volume
    
    risk_pct = (risk_per_unit / entry_price) * 100
    reward_pct = (reward_per_unit / entry_price) * 100
    
    risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
    
    return {
        "risk_per_unit": risk_per_unit,
        "reward_per_unit": reward_per_unit,
        "total_risk": total_risk,
        "total_reward": total_reward,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "risk_reward_ratio": risk_reward_ratio
    }

# Mocking place_kraken_order since it's a dependency
def place_kraken_order(client, pair, type, ordertype, volume, price=None, dry_run=False):
    if dry_run:
        return {"success": True, "dry_run": True}
    return {"success": True, "order": {"txid": ["ENTRY_TXID_123"]}}

def place_bracket_order(client, pair: str, side: str, volume: float, 
                       entry_price: float = None, 
                       stop_loss_price: float = None, 
                       take_profit_price: float = None,
                       dry_run: bool = False) -> dict:
    """
    Place a bracket order: entry + stop-loss + take-profit.
    """
    try:
        if not client:
            return {"error": "Client not initialized"}
        
        # Determine entry order type
        entry_ordertype = "market" if entry_price is None else "limit"
        
        # Validate bracket prices (skip if market entry, use current market price estimate)
        if entry_price:
            validation = validate_bracket_prices(side, entry_price, stop_loss_price, take_profit_price)
            if not validation["valid"]:
                return {"error": validation["error"]}
        
        if dry_run:
            # Validate all parameters
            if volume <= 0:
                return {"error": "Volume must be greater than 0"}
            if entry_ordertype == "limit" and not entry_price:
                return {"error": "Entry price required for limit orders"}
            if not stop_loss_price or stop_loss_price <= 0:
                return {"error": "Valid stop-loss price required"}
            if not take_profit_price or take_profit_price <= 0:
                return {"error": "Valid take-profit price required"}
            
            return {
                "success": True,
                "dry_run": True,
                "message": "Bracket order validated successfully",
                "entry_type": entry_ordertype,
                "stop_loss_type": "stop-loss-limit",
                "take_profit_type": "take-profit-limit"
            }
        
        results = {
            "entry_order": None,
            "stop_loss_order": None,
            "take_profit_order": None,
            "errors": []
        }
        
        # 1. Place entry order
        entry_result = place_kraken_order(
            client,
            pair=pair,
            type=side,
            ordertype=entry_ordertype,
            volume=volume,
            price=entry_price,
            dry_run=False
        )
        
        if "error" in entry_result:
            return {"error": f"Entry order failed: {entry_result['error']}"}
        
        results["entry_order"] = entry_result
        
        # 2. Place stop-loss order
        stop_offset = abs(stop_loss_price * 0.005)
        stop_limit_price = stop_loss_price - stop_offset if side == "buy" else stop_loss_price + stop_offset
        
        try:
            stop_side = "sell" if side == "buy" else "buy"
            stop_params = {
                "pair": pair,
                "type": stop_side,
                "ordertype": "stop-loss-limit",
                "volume": str(volume),
                "price": str(stop_loss_price),
                "price2": str(stop_limit_price)
            }
            stop_result = client.add_order(**stop_params)
            results["stop_loss_order"] = stop_result
        except Exception as e:
            results["errors"].append(f"Stop-loss error: {str(e)}")
        
        # 3. Place take-profit order
        try:
            tp_side = "sell" if side == "buy" else "buy"
            tp_params = {
                "pair": pair,
                "type": tp_side,
                "ordertype": "take-profit-limit",
                "volume": str(volume),
                "price": str(take_profit_price),
                "price2": str(take_profit_price)
            }
            tp_result = client.add_order(**tp_params)
            results["take_profit_order"] = tp_result
        except Exception as e:
            results["errors"].append(f"Take-profit error: {str(e)}")
        
        # Determine overall success
        if results["entry_order"] and results["stop_loss_order"] and results["take_profit_order"]:
            results["success"] = True
            results["message"] = "All three orders placed successfully!"
        elif results["entry_order"] and results["stop_loss_order"]:
            results["success"] = True
            results["message"] = "Entry and stop-loss placed. Take-profit failed - please place manually!"
        elif results["entry_order"]:
            results["success"] = True
            results["message"] = "⚠️ WARNING: Only entry order placed! Stop-loss and take-profit failed - PLACE MANUALLY IMMEDIATELY!"
        else:
            results["success"] = False
            results["message"] = "Bracket order failed"
        
        return results
    except Exception as e:
        return {"error": str(e)}

# --- Unit Tests ---

class TestBracketOrder(unittest.TestCase):
    
    def test_validate_prices_buy(self):
        # Valid buy: SL < Entry < TP
        res = validate_bracket_prices("buy", 50000, 48000, 55000)
        self.assertTrue(res["valid"])
        
        # Invalid buy: SL > Entry
        res = validate_bracket_prices("buy", 50000, 51000, 55000)
        self.assertFalse(res["valid"])
        self.assertIn("stop-loss must be BELOW", res["error"])
        
        # Invalid buy: TP < Entry
        res = validate_bracket_prices("buy", 50000, 48000, 49000)
        self.assertFalse(res["valid"])
        self.assertIn("take-profit must be ABOVE", res["error"])

    def test_validate_prices_sell(self):
        # Valid sell: SL > Entry > TP
        res = validate_bracket_prices("sell", 50000, 52000, 45000)
        self.assertTrue(res["valid"])
        
        # Invalid sell: SL < Entry
        res = validate_bracket_prices("sell", 50000, 49000, 45000)
        self.assertFalse(res["valid"])
        self.assertIn("stop-loss must be ABOVE", res["error"])
        
        # Invalid sell: TP > Entry
        res = validate_bracket_prices("sell", 50000, 52000, 51000)
        self.assertFalse(res["valid"])
        self.assertIn("take-profit must be BELOW", res["error"])

    def test_calculate_metrics(self):
        # Buy 1 unit at 100, SL 90, TP 120
        metrics = calculate_bracket_metrics(100, 90, 120, 1)
        self.assertEqual(metrics["risk_per_unit"], 10)
        self.assertEqual(metrics["reward_per_unit"], 20)
        self.assertEqual(metrics["risk_reward_ratio"], 2.0)
        self.assertEqual(metrics["risk_pct"], 10.0)
        self.assertEqual(metrics["reward_pct"], 20.0)

    def test_place_bracket_order_dry_run(self):
        mock_client = MagicMock()
        res = place_bracket_order(mock_client, "XBTUSD", "buy", 0.1, 50000, 48000, 55000, dry_run=True)
        self.assertTrue(res["success"])
        self.assertTrue(res["dry_run"])

    def test_place_bracket_order_real_mock(self):
        mock_client = MagicMock()
        # Mock add_order return values for SL and TP
        mock_client.add_order.side_effect = [
            {"result": {"txid": ["SL_TXID_456"]}}, # SL response
            {"result": {"txid": ["TP_TXID_789"]}}  # TP response
        ]
        
        res = place_bracket_order(mock_client, "XBTUSD", "buy", 0.1, 50000, 48000, 55000, dry_run=False)
        
        self.assertTrue(res["success"])
        self.assertEqual(res["message"], "All three orders placed successfully!")
        self.assertIsNotNone(res["entry_order"])
        self.assertIsNotNone(res["stop_loss_order"])
        self.assertIsNotNone(res["take_profit_order"])
        
        # Verify calls
        # Entry order is handled by place_kraken_order mock
        # SL and TP calls:
        self.assertEqual(mock_client.add_order.call_count, 2)
        
        # Check SL params
        sl_call_args = mock_client.add_order.call_args_list[0][1]
        self.assertEqual(sl_call_args["type"], "sell") # Opposite of buy
        self.assertEqual(sl_call_args["ordertype"], "stop-loss-limit")
        
        # Check TP params
        tp_call_args = mock_client.add_order.call_args_list[1][1]
        self.assertEqual(tp_call_args["type"], "sell") # Opposite of buy
        self.assertEqual(tp_call_args["ordertype"], "take-profit-limit")

if __name__ == '__main__':
    unittest.main()
