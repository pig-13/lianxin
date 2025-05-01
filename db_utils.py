
import sqlite3

DB_PATH = "muichiro_bot.db"

def get_character_by_user_id(user_id):
    conn = sqlite3.connect("muichiro_bot.db")
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

