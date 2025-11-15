# backend/seed_db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import random
from main import EnergyReading, Base
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'energy_data.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

db = SessionLocal()

# insert 7 days × 24 hours = 168 random hourly readings
now = datetime.utcnow()
for i in range(168):
    db.add(EnergyReading(
        timestamp = now - timedelta(hours=i),
        power = round(random.uniform(0.8, 2.5), 2)
    ))

db.commit()
db.close()
print("✅ 168 energy readings inserted successfully!")
