import os
import pandas as pd
import mysql.connector
from typing import Annotated, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fastapi import FastAPI, HTTPException, status, Depends
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
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ============================================
# 2. CONFIGURACIÓN DE MYSQL
# ============================================
db_config = {
    'host': os.getenv("MYSQL_HOST"),
    'port': os.getenv("MYSQL_PORT"),
    'user': os.getenv("MYSQL_USER"),
    'password': os.getenv("MYSQL_PASS"),
    'database': os.getenv("MYSQL_DB")
}

# ============================================
# 3. CREACIÓN DE API + CONFIG CORS
# ============================================
app = FastAPI(title="API Boot Project Extended ML + Auth")

origins = ["*"]

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
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print("ERROR DB:", err)
        raise HTTPException(500, "Error de conexión MySQL")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except:
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
    except:
        raise exception

# ============================================
# 8. CARGA DE DATOS + ML
# ============================================
def load_data_from_db():
    try:
        conn = mysql.connector.connect(**db_config)
        df = pd.read_sql("SELECT * FROM history", conn)
        conn.close()

        if df.empty:
            return pd.DataFrame()

        df = df.dropna(subset=["user_id", "query_text"])
        df["query_text"] = df["query_text"].astype(str).str.strip()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hora"] = df["timestamp"].dt.hour

        mapping = {"1": "promociones", "2": "juegos_extremos", "3": "restaurantes"}
        df["topic"] = df["query_text"].map(mapping).fillna("otros")

        return df
    except Exception as e:
        print("ERROR LOAD:", e)
        return pd.DataFrame()

@app.on_event("startup")
def train_models():
    print("Entrenando modelos...")
    df = load_data_from_db()
    if df.empty:
        print("No hay datos para entrenar.")
        return

    global_models["full_data"] = df

    tfidf = TfidfVectorizer(ngram_range=(1,2))
    X = tfidf.fit_transform(df["topic"])
    y = df["topic"]

    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X, y)
    global_models["tfidf_vectorizer"] = tfidf
    global_models["log_reg"] = log_reg

    pivot = df.groupby(["user_id", "topic"]).size().unstack(fill_value=0)
    if len(pivot) >= 3:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pivot)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        pivot["cluster"] = kmeans.fit_predict(scaled)

        pca = PCA(n_components=2)
        pivot[["x", "y"]] = pca.fit_transform(scaled)

        global_models["kmeans"] = kmeans
        global_models["scaled_users"] = pivot.reset_index()
        global_models["pca"] = pca

    print("Modelos entrenados ✔")

# ============================================
# 9. ENDPOINTS DE AUTENTICACIÓN
# ============================================
@app.post("/register")
def register_user(user: UserCreate):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT username FROM users WHERE username=%s", (user.username,))
    if cursor.fetchone():
        raise HTTPException(400, "El usuario ya existe")

    hashed = hash_password(user.password)

    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s,%s)",
                       (user.username, hashed))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(500, f"Error registrando usuario: {e}")

    cursor.close()
    conn.close()
    return {"message": "Usuario registrado"}

@app.post("/login", response_model=Token)
def login(user: UserCreate):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT id, password_hash FROM users WHERE username=%s", (user.username,))
    db_user = cursor.fetchone()

    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(401, "Credenciales inválidas")

    token = create_access_token(
        data={"sub": user.username, "user_id": db_user["id"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": token, "token_type": "bearer", "user_id": db_user["id"]}

# ============================================
# 10. ENDPOINT nuevo: registrar historial
# ============================================
@app.post("/history/record")
def record_history(record: HistoryRecord, current: TokenData = Depends(get_current_user)):
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
        raise HTTPException(500, f"Error guardando historial: {e}")

    cursor.close()
    conn.close()

    train_models()  # en producción lo ideal es async

    return {"message": "Historial guardado"}

# ============================================
# 11. ENDPOINT raíz
# ============================================
@app.get("/")
def home():
    return {"message": "API ML + Auth funcionando en Render ✔"}

# ============================================
# 12. RUN (Render usa PORT dinámico)
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
