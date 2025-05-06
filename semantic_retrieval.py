import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ====== 基本設定 ======
DB_PATH = "muichiro_bot.db"
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
id_mapping = []

# ====== 建立記憶向量索引 ======
def build_index(user_id=None):
    global id_mapping, index
    index.reset()
    id_mapping = []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if user_id:
        cursor.execute("""
            SELECT id, embedding FROM memories
            WHERE embedding IS NOT NULL AND role = 'memory' AND user_id = ?
        """, (user_id,))
    else:
        cursor.execute("""
            SELECT id, embedding FROM memories
            WHERE embedding IS NOT NULL AND role = 'memory'
        """)

    rows = cursor.fetchall()
    embeddings = []

    for mem_id, emb_blob in rows:
        vec = np.frombuffer(emb_blob, dtype=np.float32)
        embeddings.append(vec)
        id_mapping.append(mem_id)

    if embeddings:
        index.add(np.array(embeddings))

    conn.close()
    print(f"✅ FAISS 索引初始化完成，載入 {len(id_mapping)} 筆記憶")

# ====== 查詢語意相似記憶 ======
def get_similar_memories(query_text, top_k=3, max_distance=None):
    if index.ntotal == 0:
        print("⚠️ FAISS 索引為空，請先呼叫 build_index()")
        return []

    # 1. 取得距離與索引位置
    query_vec = model.encode([query_text])[0].astype(np.float32)
    D, I = index.search(np.array([query_vec]), min(top_k, index.ntotal))

    print("\n🧪 原始搜尋結果（含所有距離）:")
    for pos, dist in zip(I[0], D[0]):
        print(f" - ID {id_mapping[pos]}, 距離：{dist:.4f}")

    # 2. 把「位置」換成「真實 id」再過濾
    filtered = []
    for pos, dist in zip(I[0], D[0]):
        if pos < len(id_mapping):
            mem_id = id_mapping[pos]
            if max_distance is None or dist < max_distance:
                filtered.append((mem_id, dist))

    if not filtered:
        print("⚠️ 沒有找到符合距離條件的記憶")
        return []

    # 3. 取出內容
    similar_ids = [mem_id for mem_id, _ in filtered]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT id, content FROM memories
        WHERE id IN ({','.join('?' * len(similar_ids))})
    """, similar_ids)
    rows = cursor.fetchall()
    conn.close()

    id_to_content = {i: c for i, c in rows}
    ordered = [(id_to_content[mem_id], dist)
               for mem_id, dist in filtered if mem_id in id_to_content]
    return ordered

# ====== 測試區塊 ======
if __name__ == "__main__":
    test_user = "545532372758560769"
    test_input = "劍道社"

    build_index(test_user)
    results = get_similar_memories(test_input, top_k=5, max_distance=1.4)

    print("\n🔍 找到的相似記憶（含距離）：")
    for text, dist in results:
        print(f" - 距離 {dist:.4f}：{text}")
