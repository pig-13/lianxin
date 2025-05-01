import sqlite3

DB_PATH = "muichiro_bot.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 建立 characters 表
cursor.execute("""
CREATE TABLE IF NOT EXISTS characters (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    age TEXT,
    occupation TEXT,
    relationship TEXT,
    background TEXT,
    personality TEXT,
    speaking_style TEXT,
    likes TEXT,
    dislikes TEXT,
    extra TEXT
);
""")

# 建立 memories 表
cursor.execute("""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    role TEXT,
    content TEXT
);
""")

# 建立 summary_counts 表
cursor.execute("""
CREATE TABLE IF NOT EXISTS summary_counts (
    user_id TEXT PRIMARY KEY,
    count INTEGER
);
""")

# 建立 reminders 表（用於提醒功能）
cursor.execute("""
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    scheduled TEXT,
    reminder_text TEXT
);
""")

conn.commit()
conn.close()

print("✅ 資料庫已初始化完成，所有表格建立成功！")
