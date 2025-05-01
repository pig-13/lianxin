import sqlite3
import re

DB_PATH = "muichiro_bot.db"
MEMORY_FILE = "memory.txt"
USER_ID = "545532372758560769"  # ← 請改成你自己的 Discord user ID

# 讀取文字檔
with open(MEMORY_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 使用正規表達式分段（保留標題）
pattern = r"(【記憶\d+】\s*\d{4}-\d{1,2}-\d{1,2})\n"
parts = re.split(pattern, content)

# 整理為 (role, content)
memories = []
for i in range(1, len(parts), 2):
    header = parts[i].strip()
    body = parts[i+1].strip()
    if body:
        # header 不存進去，只保留內容即可
        memories.append(("user", body))

# 寫入 SQLite 資料庫
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 清空舊資料（可選）
cursor.execute("DELETE FROM memories WHERE user_id = ?", (USER_ID,))

# 插入新資料
for role, content in memories:
    cursor.execute(
        "INSERT INTO memories (user_id, role, content) VALUES (?, ?, ?)",
        (USER_ID, role, content)
    )

conn.commit()
conn.close()

print(f"✅ 已成功將 {len(memories)} 筆記憶寫入 SQLite！")
