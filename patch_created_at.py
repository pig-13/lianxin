import sqlite3, re, datetime as dt

DB = "lianxin_ai.db"
# 支援 2025-2-15、2025/02/15、2025年2月15日
DATE_RE = re.compile(r"(\d{4})[-/年]\s*(\d{1,2})[-/月]\s*(\d{1,2})")

with sqlite3.connect(DB) as conn:
    cur = conn.cursor()
    cur.execute("""
        SELECT id, content
        FROM memories
        WHERE role='memory'
    """)
    rows = cur.fetchall()

    updated = skipped = 0
    for _id, text in rows:
        m = DATE_RE.search(text)
        if m:
            y, mth, d = map(int, m.groups())
            date_str = f"{y:04d}-{mth:02d}-{d:02d} 00:00:00"
            cur.execute("UPDATE memories SET created_at=? WHERE id=?",
                        (date_str, _id))
            updated += 1
        else:
            skipped += 1            # 真的抓不到就先略過

    conn.commit()

print(f"✅ 已寫入 {updated} 筆日期，⚠️ 跳過 {skipped} 筆無日期")
