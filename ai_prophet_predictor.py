import pandas as pd
from prophet import Prophet
from pymongo import MongoClient
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv

# قفل رسايل التحذير المزعجة عشان الـ Console يبقى نضيف
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# الاتصال بالداتا بيز
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["project_management"]

# إعدادات الموديل
MIN_HISTORY_RECORDS = 5   # الحد الأدنى للسجلات عشان Prophet يتدرب
FORECAST_DAYS = 14        # كام يوم قدام هيتوقع
ALERT_THRESHOLD_DAYS = 10 # لو المخزن هيخلص في الفترة دي، نفتح تيكت

def prepare_daily_dataframe(history):
    """
    Converts MongoDB records into a Prophet-ready DataFrame.
    Aggregates withdrawn quantities per day into a single row.
    """
    df = pd.DataFrame(history)
    df["ds"] = pd.to_datetime(df["transaction_date"]).dt.normalize()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    
    daily_df = df.groupby("ds")["quantity"].sum().reset_index()
    daily_df.rename(columns={"quantity": "y"}, inplace=True)
    return daily_df

def train_and_forecast(daily_df):
    """
    Builds, trains a Prophet model and returns the forecast.
    """
    if daily_df["ds"].nunique() < 2:
        return None

    model = Prophet(
        yearly_seasonality=False, 
        weekly_seasonality=True,  
        daily_seasonality=False
    )
    model.fit(daily_df)

    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)
    return forecast

def create_alert_ticket(item_name, current_qty, days_left, empty_date):
    """
    Opens an emergency ticket if stock is about to run out.
    """
    ticket_title = f"AI Stock Alert: {item_name}"

    # التعديل: db.tickets وكلمة open سمول
    existing = db.tickets.find_one({"name": ticket_title, "status": "open"})
    if existing:
        print(f"   ⏳ An open alert ticket already exists for this item. Skipping.")
        return

    # التعديل: db.users و role
    admin = db.users.find_one({"role": "admin"})
    admin_id = admin["_id"] if admin else None

    db.tickets.insert_one({
        "name": ticket_title,
        "description": (
            f"AI System Warning: Current stock for '{item_name}' is ({current_qty}) units. "
            f"Based on the predicted consumption rate, stock is expected to run out in {days_left} days "
            f"(by {empty_date.strftime('%Y-%m-%d')}). Please contact suppliers immediately."
        ),
        "priority": "high", # سمول
        "status": "open", # سمول
        "created_by": admin_id,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    })
    print(f"   🎟️ Emergency ticket created automatically.")

def analyze_stock_item(stock):
    item_id = stock["_id"]
    item_name = stock.get("name", "Unknown Item")
    current_qty = stock.get("quantity", 0)

    print(f"\n{'='*40}")
    print(f"📦 Analyzing: '{item_name}' | Current stock: {current_qty}")

    if current_qty <= 0:
        print(f"   ⚠️ Stock is already empty.")
        return

    history = list(db.ai_stock_history.find({"stock_id": item_id, "action": "remove"}))

    if len(history) < MIN_HISTORY_RECORDS:
        print(f"   📊 Insufficient data for training ({len(history)}/{MIN_HISTORY_RECORDS} records).")
        return

    print(f"   🧠 Training Meta Prophet model on {len(history)} records...")
    daily_df = prepare_daily_dataframe(history)
    forecast = train_and_forecast(daily_df)

    if forecast is None:
        print(f"   ⚠️ All records share the same date. Prophet requires variation across dates.")
        return

    predicted_daily_burn = forecast.tail(FORECAST_DAYS)["yhat"].mean()
    daily_burn_rate = max(0.1, predicted_daily_burn)
    
    days_left = int(current_qty / daily_burn_rate)
    empty_date = datetime.utcnow() + timedelta(days=days_left)

    print(f"   ➤ Predicted daily burn rate: {round(daily_burn_rate, 2)} units/day")
    print(f"   ➤ Stock expected to run out in: {days_left} days (by {empty_date.strftime('%Y-%m-%d')})")

    if days_left <= ALERT_THRESHOLD_DAYS:
        print(f"   🚨 Critical warning: stock is about to run out!")
        create_alert_ticket(item_name, current_qty, days_left, empty_date)
    else:
        print(f"   ✅ Stock level is safe.")

def predict_stock_with_meta():
    print("🤖 Meta Prophet AI Core Started...\n" + "="*40)
    
    # التعديل: db.stockitems
    stocks = list(db.stockitems.find())

    if not stocks:
        print("❌ No stock items found in the database.")
        return

    for stock in stocks:
        analyze_stock_item(stock)

    print(f"\n{'='*40}")
    print("🏁 Prediction cycle completed successfully.")

if __name__ == "__main__":
    predict_stock_with_meta()