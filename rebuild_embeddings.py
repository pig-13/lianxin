import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "muichiro_bot.db"
model = SentenceTransformer('all-MiniLM-L6-v2')

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 查詢還沒 embedding 的記憶
cursor.execute("""
    SELECT id, content FROM memories
    WHERE embedding IS NULL AND role = 'memory'
""")
rows = cursor.fetchall()
print(f"🔍 找到 {len(rows)} 筆需要轉換的記憶")

count = 0
for mem_id, text in rows:
    try:
        vec = model.encode([text])[0].astype(np.float32)
        cursor.execute("UPDATE memories SET embedding = ? WHERE id = ?", (vec.tobytes(), mem_id))
        count += 1
    except Exception as e:
        print(f"❌ 第 {mem_id} 筆轉換失敗：{e}")

conn.commit()
conn.close()
print(f"✅ 完成轉換，共寫入 {count} 筆記憶嵌入向量")
