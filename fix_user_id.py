import sqlite3

DB_PATH = "lianxin_ai.db"

WRONG_ID = "123456789012345678"
CORRECT_ID = "545532372758560769"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    UPDATE memories
    SET user_id = ?
    WHERE user_id = ?
""", (CORRECT_ID, WRONG_ID))

affected = cursor.rowcount
conn.commit()
conn.close()

print(f"✅ 修正完成，共更新 {affected} 筆記憶 user_id → {CORRECT_ID}")
