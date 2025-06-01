
import sqlite3
import sys
import os

# ✅ 加入這段：讓打包後也能找到正確資料庫路徑
def get_resource_path(relative_path):
    """打包與未打包都可用的資源定位方法"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

DB_PATH = get_resource_path("lianxin_ai.db")

def get_character_by_user_id(user_id):
    conn = sqlite3.connect(DB_PATH)  # ✅ 改這裡
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM characters WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "name": row[1],
        "age": row[2],
        "occupation": row[3],
        "relationship": row[4],
        "background": row[5],
        "personality": row[6],
        "speaking_style": row[7],
        "likes": row[8],
        "dislikes": row[9],
        "extra": row[10],
    }

def get_latest_memory_id():
    conn = sqlite3.connect("DB_PATH")
    cur = conn.cursor()
    cur.execute("SELECT MAX(id) FROM memories WHERE role='memory'")
    result = cur.fetchone()
    conn.close()
    return result[0] if result and result[0] is not None else 0
