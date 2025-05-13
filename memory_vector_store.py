from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sqlite3
import datetime
from init_db import ensure_schema
ensure_schema()

# 初始化嵌入模型（固定維度 384）
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)

DB_PATH = "lianxin_ai.db"

def add_memory(user_id, role, content, importance=0):
    # 1. 轉向量
    embedding = model.encode([content])[0].astype(np.float32)
    embedding_bytes = embedding.tobytes()

    # 2. 存到 SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memories (user_id, role, content, created_at, importance, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, role, content, datetime.datetime.now().isoformat(), importance, embedding_bytes))
    conn.commit()

    # 3. 加入 FAISS
    index.add(np.array([embedding]))

    conn.close()
    print("✅ 成功加入記憶到資料庫與 FAISS")

# 範例測試
if __name__ == "__main__":
    add_memory("test_user", "memory", "她說最喜歡夏天的陽光。")
