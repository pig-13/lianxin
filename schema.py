import sqlite3

DB_PATH = "muichiro_bot.db"  # 你的資料庫名稱

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 列出所有表格名稱
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("📋 所有表格：")
for (table_name,) in tables:
    print(f"\n🧾 表格：{table_name}")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for col in columns:
        cid, name, ctype, notnull, default, pk = col
        print(f"  - {name} ({ctype}) {'[PK]' if pk else ''}")
