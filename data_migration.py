"""
Data Migration Script for TraderQ
Migrates data from local JSON files to Firebase Firestore

Usage:
    python data_migration.py

This script will:
1. Check for Firebase setup
2. Prompt for user email (must be an existing Firebase user)
3. Read all JSON files
4. Migrate data to Firestore
5. Create backup of JSON files
"""

import json
import os
from pathlib import Path
from datetime import datetime
import firebase_admin
import argparse
from firebase_admin import credentials, auth, firestore
from firebase_db import FirestoreDB


# JSON file paths
BASE_DIR = Path(__file__).parent
CUSTOM_TICKERS_FILE = BASE_DIR / ".custom_tickers.json"
ALERTS_FILE = BASE_DIR / ".alerts.json"
PORTFOLIO_FILE = BASE_DIR / ".portfolio.json"
CROSS_HISTORY_FILE = BASE_DIR / ".cross_history.json"
WATCHLISTS_FILE = BASE_DIR / ".watchlists.json"
TRADE_JOURNAL_FILE = BASE_DIR / ".trade_journal.json"
ALERT_HISTORY_FILE = BASE_DIR / ".alert_history.json"
EMAIL_CONFIG_FILE = BASE_DIR / ".email_config.json"
KRAKEN_CONFIG_FILE = BASE_DIR / ".kraken_config.json"


def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            service_account_path = os.getenv(
                'GOOGLE_APPLICATION_CREDENTIALS',
                './serviceAccountKey.json'
            )

            if not os.path.exists(service_account_path):
                print(f"❌ Error: Service account key not found at {service_account_path}")
                print("Please follow FIREBASE_SETUP.md to download your service account key.")
                return False

            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized successfully")
            return True
    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")
        return False


def get_user_id_by_email(email: str) -> str:
    """Get Firebase user ID from email"""
    try:
        user = auth.get_user_by_email(email)
        return user.uid
    except auth.UserNotFoundError:
        print(f"❌ User not found: {email}")
        print("Please create an account first using the TraderQ app.")
        return None
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return None


def load_json_file(filepath: Path) -> dict | list:
    """Load JSON file if it exists"""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load {filepath.name}: {e}")
            return None
    return None


def backup_json_files():
    """Create backup of all JSON files"""
    backup_dir = BASE_DIR / "json_backup"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = backup_dir / f"backup_{timestamp}"
    backup_subdir.mkdir(exist_ok=True)

    files = [
        CUSTOM_TICKERS_FILE, ALERTS_FILE, PORTFOLIO_FILE,
        CROSS_HISTORY_FILE, WATCHLISTS_FILE, TRADE_JOURNAL_FILE,
        ALERT_HISTORY_FILE, EMAIL_CONFIG_FILE, KRAKEN_CONFIG_FILE
    ]

    backed_up = 0
    for file in files:
        if file.exists():
            try:
                import shutil
                shutil.copy2(file, backup_subdir / file.name)
                backed_up += 1
            except Exception as e:
                print(f"⚠️  Warning: Could not backup {file.name}: {e}")

    print(f"✅ Backed up {backed_up} files to {backup_subdir}")
    return backup_subdir


def migrate_data(user_id: str):
    """Migrate all JSON data to Firestore"""
    db = FirestoreDB()
    migrated_items = []

    print(f"\n🚀 Starting migration for user {user_id}...\n")

    # 1. Migrate Custom Tickers
    custom_tickers = load_json_file(CUSTOM_TICKERS_FILE)
    if custom_tickers:
        try:
            db.save_custom_tickers(user_id, custom_tickers)
            # Also save selected tickers if present
            if 'selected' in custom_tickers:
                db.save_selected_tickers(user_id, custom_tickers['selected'])
            print("✅ Migrated custom tickers")
            migrated_items.append("Custom Tickers")
        except Exception as e:
            print(f"❌ Failed to migrate custom tickers: {e}")

    # 2. Migrate Alerts
    alerts = load_json_file(ALERTS_FILE)
    if alerts and isinstance(alerts, list):
        try:
            db.save_alerts(user_id, alerts)
            print(f"✅ Migrated {len(alerts)} alerts")
            migrated_items.append(f"{len(alerts)} Alerts")
        except Exception as e:
            print(f"❌ Failed to migrate alerts: {e}")

    # 3. Migrate Portfolio
    portfolio = load_json_file(PORTFOLIO_FILE)
    if portfolio:
        try:
            db.save_portfolio(user_id, portfolio)
            print("✅ Migrated portfolio")
            migrated_items.append("Portfolio")
        except Exception as e:
            print(f"❌ Failed to migrate portfolio: {e}")

    # 4. Migrate Watchlists
    watchlists = load_json_file(WATCHLISTS_FILE)
    if watchlists and isinstance(watchlists, dict):
        try:
            db.save_watchlists(user_id, watchlists)
            print(f"✅ Migrated {len(watchlists)} watchlists")
            migrated_items.append(f"{len(watchlists)} Watchlists")
        except Exception as e:
            print(f"❌ Failed to migrate watchlists: {e}")

    # 5. Migrate Trade Journal
    trade_journal = load_json_file(TRADE_JOURNAL_FILE)
    if trade_journal and isinstance(trade_journal, list):
        try:
            db.save_trade_journal(user_id, trade_journal)
            print(f"✅ Migrated {len(trade_journal)} trades")
            migrated_items.append(f"{len(trade_journal)} Trades")
        except Exception as e:
            print(f"❌ Failed to migrate trade journal: {e}")

    # 6. Migrate Cross History
    cross_history = load_json_file(CROSS_HISTORY_FILE)
    if cross_history and isinstance(cross_history, dict):
        try:
            db.save_cross_history(user_id, cross_history)
            print(f"✅ Migrated cross history for {len(cross_history)} tickers")
            migrated_items.append(f"Cross History ({len(cross_history)} tickers)")
        except Exception as e:
            print(f"❌ Failed to migrate cross history: {e}")

    # 7. Migrate Alert History
    alert_history = load_json_file(ALERT_HISTORY_FILE)
    if alert_history and isinstance(alert_history, list):
        try:
            for item in alert_history:
                db.add_alert_history(user_id, item)
            print(f"✅ Migrated {len(alert_history)} alert history items")
            migrated_items.append(f"{len(alert_history)} Alert History items")
        except Exception as e:
            print(f"❌ Failed to migrate alert history: {e}")

    # 8. Migrate Email Config (temporary - will move to Secret Manager later)
    email_config = load_json_file(EMAIL_CONFIG_FILE)
    if email_config:
        try:
            db.save_email_config(user_id, email_config)
            print("✅ Migrated email config")
            migrated_items.append("Email Config")
            print("⚠️  NOTE: Email passwords stored in Firestore temporarily.")
            print("   Will be moved to Secret Manager in Phase 2.")
        except Exception as e:
            print(f"❌ Failed to migrate email config: {e}")

    # 9. Migrate Kraken Config (temporary - will move to Secret Manager later)
    kraken_config = load_json_file(KRAKEN_CONFIG_FILE)
    if kraken_config:
        try:
            db.save_kraken_config(user_id, kraken_config)
            print("✅ Migrated Kraken API config")
            migrated_items.append("Kraken API Config")
            print("⚠️  NOTE: API keys stored in Firestore temporarily.")
            print("   Will be moved to Secret Manager in Phase 2.")
        except Exception as e:
            print(f"❌ Failed to migrate Kraken config: {e}")

    return migrated_items


def main():
    """Main migration script"""
    parser = argparse.ArgumentParser(description='TraderQ Data Migration')
    parser.add_argument('--email', help='Firebase account email')
    parser.add_argument('--yes', '-y', action='store_true', help='Confirm migration automatically')
    args = parser.parse_args()

    print("=" * 60)
    print("TraderQ Data Migration Script")
    print("Migrate JSON files → Firebase Firestore")
    print("=" * 60)
    print()

    # Initialize Firebase
    if not initialize_firebase():
        return

    # Get user email
    if args.email:
        email = args.email
        print(f"\n📧 Using email from argument: {email}")
    else:
        print("\n📧 Enter your Firebase account email")
        print("   (This should be the email you used to sign up in TraderQ)")
        email = input("Email: ").strip()

    if not email:
        print("❌ No email provided. Exiting.")
        return

    # Get user ID
    print(f"\n🔍 Looking up user: {email}...")
    user_id = get_user_id_by_email(email)

    if not user_id:
        return

    print(f"✅ Found user ID: {user_id}")

    # Confirm migration
    print("\n⚠️  This will migrate your local JSON data to Firestore.")
    print("   Your JSON files will be backed up before migration.")
    
    if args.yes:
        confirm = 'yes'
        print("\nProceed with migration? (yes/no): yes (auto-confirmed)")
    else:
        confirm = input("\nProceed with migration? (yes/no): ").strip().lower()

    if confirm not in ['yes', 'y']:
        print("Migration cancelled.")
        return

    # Backup JSON files
    print("\n📦 Creating backup of JSON files...")
    backup_dir = backup_json_files()

    # Migrate data
    migrated_items = migrate_data(user_id)

    # Summary
    print("\n" + "=" * 60)
    print("✅ MIGRATION COMPLETE!")
    print("=" * 60)
    print(f"\nMigrated {len(migrated_items)} items:")
    for item in migrated_items:
        print(f"  • {item}")

    print(f"\n📦 Backup location: {backup_dir}")
    print("\n🎯 Next Steps:")
    print("  1. Verify data in Firebase Console: https://console.firebase.google.com/")
    print("  2. Test the TraderQ app with your Firebase account")
    print("  3. If everything works, you can safely delete the JSON files")
    print("\n⚠️  Keep the backup until you've verified everything works!")
    print()


if __name__ == "__main__":
    main()
