import os
import io
import logging
import uuid
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.ensemble import GradientBoostingClassifier
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.backends.redis import RedisBackend
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
from redis import asyncio as aioredis
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

# 1. Security Configurations (In production, load these from environment variables!)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-development-key-change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], bcrypt__ident="2b", deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock User Database (For production, you'd pull this from a `users` table in PostgreSQL/SQLite)
# The password below is hashed for "password123"
# USER_DB = {
#     "admin": {
#         "username": "admin",
#         "hashed_password": pwd_context.hash("password123")
#     }
# }

# --- 1. DATABASE CONFIGURATION ---
# Render automatically provides a DATABASE_URL environment variable. 
# If running locally, it falls back to a local SQLite file so your code never breaks.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_scans.db")

# Fix for PostgreSQL URLs which sometimes start with 'postgres://' instead of 'postgresql://'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the Database Schema
class MachineryScan(Base):
    __tablename__ = "machinery_scans"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    air_temperature = Column(Float)
    process_temperature = Column(Float)
    rotational_speed = Column(Integer)
    torque = Column(Float)
    tool_wear = Column(Integer)
    failure_risk_detected = Column(Boolean)
    failure_probability_percent = Column(Float)

    # Human-in-the-Loop Feedback (Initially NULL!)
    actual_failure = Column(Integer, nullable=True) 
    reviewed_at = Column(DateTime, nullable=True)

# 💾 NEW: Add the database table schema for your authentic user base
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

# Automatically create the table structure on startup
Base.metadata.create_all(bind=engine)

# Dependency to safely handle database sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# def verify_password(plain_password, hashed_password):
#     try:
#         return pwd_context.verify(plain_password, hashed_password)
#     except Exception:
#         # Fallback check in case the server environment completely lacks bcrypt binaries
#         return plain_password == "password123"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    # 🛠️ FIXED: Changed utcnow() to standard UTC timestamp handling for modern python compliance
    expire = datetime.now(timezone.utc if 'timezone' in globals() else None) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
        # or username not in USER_DB:
            raise credentials_exception
        # return username
    except JWTError:
        raise credentials_exception

    # 🔍 REAL DB LOOKUP: Check if the user exists in the active table database lines
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user.username

# --- 2. INITIALIZE DEFAULT ADMIN ACCOUNT ---
# This checks if the user table is empty at boot time, and populates a baseline admin account
def create_initial_admin_user():
    db = SessionLocal()
    try:
        admin_exists = db.query(User).filter(User.username == "admin").first()
        if not admin_exists:
            # We hardcode the secure hash generated earlier for "password123"
            default_admin = User(
                username="admin",
                hashed_password= pwd_context.hash("passbaru")
            )
            db.add(default_admin)
            db.commit()
            print("👤 System Database: Default 'admin' user successfully initialized.")
    finally:
        db.close()

# Run the boot initializer setup check
create_initial_admin_user()

# Configure structured JSON-style logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telemetry")

# 1. Initialize the instrumentator object
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics"]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup phase ---
    # Expose the /metrics route when the application boots up
    instrumentator.expose(app, endpoint="/metrics", tags=["Telemetry"])
    redis = aioredis.from_url("redis://localhost:6379", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    yield

# --- 2. FASTAPI & ML MODEL CONFIG ---
app = FastAPI(title="Robust Failure Engine with Logging", lifespan = lifespan)
# FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")

class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Assign a unique Correlation ID to trace the request execution path
        request_id = str(uuid.uuid4())
        
        # 2. Record start timestamp
        start_time = time.perf_counter()
        
        # 3. Process the incoming HTTP request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            # Catch unhandled backend server errors (500s)
            status_code = 500
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            logger.error(
                f"[TELEMETRY] request_id={request_id} method={request.method} "
                f"path={request.url.path} status=500 latency_ms={process_time_ms} "
                f"error={str(exc)}"
            )
            raise exc

        # 4. Calculate total execution latency in milliseconds
        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 5. Inject telemetry correlation headers onto the client response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = str(process_time_ms)

        # 6. Log structured telemetry data
        log_message = (
            f"[TELEMETRY] request_id={request_id} method={request.method} "
            f"path={request.url.path} status={status_code} latency_ms={process_time_ms}"
        )
        
        if status_code >= 500:
            logger.error(log_message)
        elif status_code >= 400:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        return response

# Register middleware with FastAPI
# Define allowed frontend domains
ALLOWED_ORIGINS = [
    "http://localhost:8501",                      # Local Streamlit
    "https://appproject-s3jyjladz6h9xknvkddvcy.streamlit.app/",        # Production Streamlit Cloud URL
]

app.add_middleware(TelemetryMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Allow Streamlit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # 💥 THIS IS THE CRITICAL LINE:
    expose_headers=["X-Request-ID", "X-Process-Time-MS"], 
)

class KaggleTelemetry(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int

# (Keep your existing cached model loading logic here)
MODEL = None
csv_path = os.path.join(os.path.dirname(__file__), "predictive_maintenance.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={
        'Air temperature': 'air_temperature', 'Process temperature': 'process_temperature',
        'Rotational speed': 'rotational_speed', 'Torque': 'torque', 'Tool wear': 'tool_wear',
        'machine_failure': 'machine_failure'
    }, inplace=True)
    X = df[['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']]
    y = df['machine_failure']
    MODEL = GradientBoostingClassifier(n_estimators=100, max_depth=12, random_state=42)
    MODEL.fit(X, y)

# --- 3. ENDPOINTS ---

# 2. Login Endpoint to generate tokens
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # user = USER_DB.get(form_data.username)
    # 🔍 REAL DB LOOKUP: Query the credentials directly out of the user table
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.username}, 
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict_failure(data: KaggleTelemetry, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact is offline.")
    
    # input_df = pd.DataFrame([data.dict()])
    input_df = pd.DataFrame([data.model_dump()])
    predicted_class = int(MODEL.predict(input_df)[0])
    probabilities = MODEL.predict_proba(input_df)[0]
    failure_probability = float(probabilities[1])
    
    is_failure = (predicted_class == 1)
    prob_percent = round(failure_probability * 100, 2)
    
    importances = MODEL.feature_importances_
    risk_contributors = {
        "Thermal Stress (Air)": round(float((data.air_temperature - 295) * importances[0] * 10), 2),
        "Thermal Stress (Process)": round(float((data.process_temperature - 305) * importances[1] * 10), 2),
        "Rotational Stress Index": round(float((data.rotational_speed / 2900) * importances[2] * 100), 2),
        "Torque Strain Metric": round(float((data.torque / 77) * importances[3] * 100), 2),
        "Tool Fatigue Level": round(float((data.tool_wear / 250) * importances[4] * 100), 2),
    }
    
    # 💾 THE LOGGING MAGIC: Write the scan details straight into the database
    db_scan = MachineryScan(
        air_temperature=data.air_temperature,
        process_temperature=data.process_temperature,
        rotational_speed=data.rotational_speed,
        torque=data.torque,
        tool_wear=data.tool_wear,
        failure_risk_detected=is_failure,
        failure_probability_percent=prob_percent
    )
    db.add(db_scan)
    db.commit() # Saves permanently
    
    return {
        "failure_risk_detected": is_failure,
        "failure_probability_percent": prob_percent,
        "risk_factors": risk_contributors
    }

# 🛠️ NEW ENDPOINT: Let the frontend fetch the log history
@app.get("/history")
async def get_scan_history(limit: int = 10, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    scans = db.query(MachineryScan).order_by(MachineryScan.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": s.id,
            "timestamp": s.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "telemetry": {
                "Air Temp": s.air_temperature, "Process Temp": s.process_temperature,
                "RPM": s.rotational_speed, "Torque": s.torque, "Tool Wear": s.tool_wear
            },
            "risk_detected": s.failure_risk_detected,
            "probability": s.failure_probability_percent
        }
        for s in scans
    ]

@app.get("/analytics/summary")
async def get_fleet_summary(db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    # Calculate key metrics across the entire logged database history
    total_scans = db.query(MachineryScan).count()
    total_failures = db.query(MachineryScan).filter(MachineryScan.failure_risk_detected == True).count()
    avg_probability = db.query(func.avg(MachineryScan.failure_probability_percent)).scalar() or 0.0
    
    return {
        "total_scans": total_scans,
        "total_failures_detected": total_failures,
        "average_failure_probability": round(float(avg_probability), 2)
    }

@app.get("/analytics/stress-grid")
async def get_stress_grid(
    air_temp: float, 
    proc_temp: float, 
    tool_wear: int, 
    db: Session = Depends(get_db), 
    current_user: str = Depends(get_current_user)
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model offline.")
        
    # Generate range arrays spanning low to high operational extremes
    rpm_range = list(range(1100, 3100, 200)) # 10 steps
    torque_range = [round(t, 1) for t in np.linspace(3.0, 77.0, 10)] # 10 steps
    
    grid_records = []
    
    # Sweep combinations to simulate the failure envelope matrix
    for rpm in rpm_range:
        for torque in torque_range:
            mock_payload = {
                "air_temperature": air_temp,
                "process_temperature": proc_temp,
                "rotational_speed": rpm,
                "torque": torque,
                "tool_wear": tool_wear
            }
            input_df = pd.DataFrame([mock_payload])
            probabilities = MODEL.predict_proba(input_df)[0]
            prob_percent = round(float(probabilities[1]) * 100, 2)
            
            grid_records.append({
                "RPM": rpm,
                "Torque (Nm)": torque,
                "Failure Probability (%)": prob_percent
            })
            
    return grid_records

class UserCreate(BaseModel):
    username: str
    password: str

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    user_in: UserCreate, 
    db: Session = Depends(get_db), 
    current_user: str = Depends(get_current_user) # 🔐 FIXED: Enforces valid token to use this route
):
    username_clean = user_in.username.strip()
    
    if not username_clean or not user_in.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password fields cannot be empty."
        )
        
    # Check if the username is already taken
    existing_user = db.query(User).filter(User.username == username_clean).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This username is already registered in the system."
        )
        
    secured_hash = pwd_context.hash(user_in.password)
    new_user = User(username=username_clean, hashed_password=secured_hash)
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"message": f"User '{new_user.username}' successfully created."}
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database write failure during account provisioning."
        )

@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db), 
    current_user: str = Depends(get_current_user)
):
    # 1. Securely read the uploaded file bytes into memory
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a standard CSV.")
        
    # 2. Validate that required features exist in the uploaded spreadsheet columns
    required_cols = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing structural features in CSV columns: {missing}"
        )
        
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Predictive model offline.")
        
    # 3. Compute bulk predictions matching your existing ML pipeline structure
    features_df = df[required_cols]
    probabilities = MODEL.predict_proba(features_df)[:, 1]
    predictions = MODEL.predict(features_df)
    
    # 4. Append predictions back onto the data frame columns
    df["failure_probability_percent"] = (probabilities * 100).round(2)
    df["failure_risk_detected"] = predictions.astype(bool)
    
    # 5. Build list of DB model instances for highly efficient bulk insertion
    db_records = []
    for _, row in df.iterrows():
        record = MachineryScan(
            air_temperature=float(row["air_temperature"]),
            process_temperature=float(row["process_temperature"]),
            rotational_speed=int(row["rotational_speed"]),
            torque=float(row["torque"]),
            tool_wear=int(row["tool_wear"]),
            failure_risk_detected=bool(row["failure_risk_detected"]),
            failure_probability_percent=float(row["failure_probability_percent"])
        )
        db_records.append(record)
        
    try:
        db.bulk_save_objects(db_records)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database bulk insertion transaction failed.")
        
    # 6. Stream the results directly back to the client as an annotated CSV string stream
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_predictions.csv"}
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}

class FeedbackUpdate(BaseModel):
    prediction_id: int
    actual_failure: int  # 0 for normal, 1 for actual failure

@app.post("/feedback")
async def log_ground_truth(feedback: FeedbackUpdate, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    # 1. Fetch record by ID
    record = db.query(MachineryScan).filter(MachineryScan.id == feedback.prediction_id).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found")
        
    # 2. Update the record with actual ground truth
    record.actual_failure = feedback.actual_failure
    record.reviewed_at = datetime.now(timezone.utc)
    
    db.commit()
    return {"status": "success", "message": "Ground truth logged for future model retraining!"}

# class MachineryScan(Base):
#     __tablename__ = "machinery_scans"
    
#     id = Column(Integer, primary_key=True, index=True)
#     timestamp = Column(DateTime, default=datetime.utcnow)
#     air_temperature = Column(Float)
#     process_temperature = Column(Float)
#     rotational_speed = Column(Integer)
#     torque = Column(Float)
#     tool_wear = Column(Integer)
#     failure_risk_detected = Column(Boolean)
#     failure_probability_percent = Column(Float)

#     # Human-in-the-Loop Feedback (Initially NULL!)
#     actual_failure = Column(Integer, nullable=True) 
#     reviewed_at = Column(DateTime, nullable=True)

@app.get("/unreviewed-predictions")
def get_unrev_pred(db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    records = db.query(MachineryScan).filter(MachineryScan.actual_failure.is_(None)).all()

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else None,
            "air_temperature": r.air_temperature,
            "process_temperature": r.process_temperature,
            "rotational_speed": r.rotational_speed,
            "torque": r.torque,
            "tool_wear": r.tool_wear,
            "failure_risk_detected": r.failure_risk_detected,
            "failure_probability_percent": r.failure_probability_percent,
            "actual_failure": r.actual_failure
        }
        for r in records
    ]
    

# BOTTOM

# Instrument app and expose /metrics endpoint
instrumentator.instrument(app)