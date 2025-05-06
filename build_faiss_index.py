import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = "muichiro_bot.db"

# 初始化模型與向量 index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)  # 384 是該模型的維度

# 開 SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 撈出所有還沒生成 embedding 的記憶
cursor.execute("SELECT id, content FROM memories WHERE embedding IS NULL AND role = 'memory'")
rows = cursor.fetchall()

print(f"🔍 找到 {len(rows)} 筆待處理記憶")

count = 0
for mem_id, content in rows:
    # 向量化
    embedding = model.encode([content])[0].astype(np.float32)
    embedding_bytes = embedding.tobytes()

    # 存回資料庫
    cursor.execute("UPDATE memories SET embedding = ? WHERE id = ?", (embedding_bytes, mem_id))

    # 同步加進 FAISS index
    index.add(np.array([embedding]))
    count += 1

conn.commit()
conn.close()

print(f"✅ 已成功轉換並同步 {count} 筆記憶向量")
