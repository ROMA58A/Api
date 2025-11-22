# main.py
import os
import pandas as pd
import mysql.connector
from typing import Annotated, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import uvicorn
import warnings
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

warnings.filterwarnings("ignore")

# ============================================
# 1. CONFIGURACIÓN DE SEGURIDAD
# ============================================
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "clave_render_segura_123")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ============================================
# 2. CONFIGURACIÓN DE MYSQL (variables en Render)
# ============================================
db_config = {
    'host': os.getenv("MYSQL_HOST", "localhost"),
    'port': int(os.getenv("MYSQL_PORT", os.getenv("DB_PORT", 3306))),
    'user': os.getenv("MYSQL_USER", "root"),
    'password': os.getenv("MYSQL_PASS", ""),
    'database': os.getenv("MYSQL_DB", "app_db"),
    'autocommit': False,
    'raise_on_warnings': True,
}

# ============================================
# 3. CREACIÓN DE API + CONFIG CORS
# ============================================
app = FastAPI(title="API Boot Project Extended ML + Auth")

origins = ["*"]  # en producción restringir al dominio

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 4. MODELOS GLOBALES
# ============================================
global_models = {
    "log_reg": None,
    "tfidf_vectorizer": None,
    "kmeans": None,
    "scaled_users": None,
    "full_data": None,
    "pca": None
}

# ============================================
# 5. Pydantic Schemas
# ============================================
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None

class Query(BaseModel):
    text: str

class HistoryRecord(BaseModel):
    query_text: str

# ============================================
# 6. UTILIDADES
# ============================================
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        return conn
    except mysql.connector.Error as err:
        print("ERROR DB:", err)
        raise HTTPException(status_code=500, detail="Error de conexión MySQL")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False

def create_access_token(data: dict, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ============================================
# 7. DEPENDENCIA JWT
# ============================================
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    exception = HTTPException(
        status_code=401,
        detail="Token inválido",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        if not username or not user_id:
            raise exception
        return TokenData(username=username, user_id=user_id)
    except JWTError:
        raise exception
    except Exception:
        raise exception

# ============================================
# 8. CARGA DE DATOS + ML
# ============================================
def load_data_from_db():
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT * FROM history", conn)
        conn.close()

        if df.empty:
            return pd.DataFrame()

        df = df.dropna(subset=["user_id", "query_text"])
        df["query_text"] = df["query_text"].astype(str).str.strip()
        # Si no existe timestamp, se maneja con coerción
        df["timestamp"] = pd.to_datetime(df.get("timestamp", pd.Series()), errors="coerce")
        if "timestamp" not in df.columns or df["timestamp"].isna().all():
            df["timestamp"] = datetime.utcnow()
        df["hora"] = pd.to_datetime(df["timestamp"]).dt.hour

        # Ejemplo mapping simple - ajustar a tus datos reales
        mapping = {"1": "promociones", "2": "juegos_extremos", "3": "restaurantes"}
        df["topic"] = df["query_text"].map(mapping).fillna("otros")

        return df
    except Exception as e:
        print("ERROR LOAD:", e)
        return pd.DataFrame()

def train_models():
    print("Entrenando modelos...")
    df = load_data_from_db()
    if df.empty or len(df) < 5:
        print("No hay suficientes datos para entrenar.")
        return

    global_models["full_data"] = df

    # TF-IDF + LogisticRegression (clasificador simple por topic)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = tfidf.fit_transform(df["topic"])
    y = df["topic"]

    log_reg = LogisticRegression(max_iter=400)
    log_reg.fit(X, y)
    global_models["tfidf_vectorizer"] = tfidf
    global_models["log_reg"] = log_reg

    # Clustering por comportamiento del usuario (pivot)
    pivot = df.groupby(["user_id", "topic"]).size().unstack(fill_value=0)
    if pivot.shape[0] >= 3 and pivot.shape[1] >= 1:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pivot)

        kmeans = KMeans(n_clusters=min(3, max(1, scaled.shape[0] // 2)), random_state=42, n_init=10)
        pivot["cluster"] = kmeans.fit_predict(scaled)

        pca = PCA(n_components=min(2, scaled.shape[1]))
        coords = pca.fit_transform(scaled)
        pivot["x"] = coords[:, 0]
        if coords.shape[1] > 1:
            pivot["y"] = coords[:, 1]
        else:
            pivot["y"] = 0.0

        global_models["kmeans"] = kmeans
        global_models["scaled_users"] = pivot.reset_index()
        global_models["pca"] = pca

    print("Modelos entrenados ✔")

# Intento entrenar al iniciar (Render ejecuta startup events)
@app.on_event("startup")
def startup_event():
    # Entrenamiento inicial no bloqueante pesado — solo intenta cargar y entrenar
    try:
        train_models()
    except Exception as e:
        print("Error en startup train:", e)

# ============================================
# 9. ENDPOINTS DE AUTENTICACIÓN
# ============================================
@app.post("/register")
def register_user(user: UserCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT username FROM users WHERE username=%s", (user.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="El usuario ya existe")

        hashed = hash_password(user.password)

        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s,%s)",
                       (user.username, hashed))
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        print("Error registrar:", e)
        raise HTTPException(status_code=500, detail=f"Error registrando usuario: {e}")
    finally:
        cursor.close()
        conn.close()

    return {"message": "Usuario registrado"}

@app.post("/login", response_model=Token)
def login(user: UserCreate):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, password_hash FROM users WHERE username=%s", (user.username,))
        db_user = cursor.fetchone()

        if not db_user or not verify_password(user.password, db_user["password_hash"]):
            raise HTTPException(status_code=401, detail="Credenciales inválidas")

        token = create_access_token(
            data={"sub": user.username, "user_id": db_user["id"]},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        return {"access_token": token, "token_type": "bearer", "user_id": db_user["id"]}
    finally:
        cursor.close()
        conn.close()

# ============================================
# 10. ENDPOINT nuevo: registrar historial
# ============================================
@app.post("/history/record")
def record_history(record: HistoryRecord, background_tasks: BackgroundTasks, current: TokenData = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO history (user_id, query_text, timestamp) VALUES (%s,%s,NOW())",
            (current.user_id, record.query_text)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("Error guardar historial:", e)
        raise HTTPException(status_code=500, detail=f"Error guardando historial: {e}")
    finally:
        cursor.close()
        conn.close()

    # Re-entrena en background para no bloquear la respuesta
    background_tasks.add_task(train_models)

    return {"message": "Historial guardado"}

# ============================================
# 11. ENDPOINTS DE UTILIDAD
# ============================================
@app.get("/")
def home():
    return {"message": "API ML + Auth funcionando en Render ✔"}

@app.get("/models/status")
def models_status():
    status = {k: (v is not None) for k, v in global_models.items()}
    return {"models": status}

# Endpoint simple para predecir topic (ejemplo)
@app.post("/predict_topic")
def predict_topic(q: Query):
    tfidf = global_models.get("tfidf_vectorizer")
    log_reg = global_models.get("log_reg")
    if tfidf is None or log_reg is None:
        raise HTTPException(status_code=503, detail="Modelos no disponibles")
    X = tfidf.transform([q.text])
    pred = log_reg.predict(X)[0]
    return {"topic": pred}

# ============================================
# 12. RUN (Render usa PORT dinámico)
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
