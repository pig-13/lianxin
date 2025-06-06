import sqlite3
import sys
import os

# ✅ 支援打包後與開發階段的路徑函式
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

DB_PATH = get_resource_path("lianxin_ai.db")

# 建立主資料庫連線
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 建立 characters 表（角色資料）
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

# 建立 user_profiles 表（使用者資料）
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    nickname TEXT,
    age TEXT,
    gender TEXT,
    background TEXT,
    extra TEXT
);
""")

# 建立 memories 表（記憶系統，含 embedding 向量欄位）
cursor.execute("""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    role TEXT,
    content TEXT,
    created_at TEXT,
    importance INTEGER,
    embedding BLOB
);
""")

# 建立 conversations 表（新增：用來儲存對話記錄）
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    user_msg TEXT,
    ai_msg TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
""")

# 建立 summary_counts 表（摘要記憶次數紀錄）
cursor.execute("""
CREATE TABLE IF NOT EXISTS summary_counts (
    user_id TEXT PRIMARY KEY,
    count INTEGER
);
""")

# 建立 reminders 表（提醒功能）
cursor.execute("""
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    scheduled TEXT,
    reminder_text TEXT,
    repeat INTEGER
);
""")

# 建立 api_log 表（API 使用次數統計）
cursor.execute("""
CREATE TABLE IF NOT EXISTS api_log (
    day_key TEXT PRIMARY KEY,
    count INTEGER
);
""")

# 若你還有分開一份記憶用的 memory.db，也一併初始化（可選用）
def init_memory_db():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character_id TEXT,
            content TEXT,
            created_at TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()
# 初始化完成

def ensure_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE memories ADD COLUMN embedding BLOB;")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise e
    conn.commit()
    conn.close()


conn.commit()
conn.close()

print("✅ 資料庫已初始化完成，所有表格與欄位建立成功！")
