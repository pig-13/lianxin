# import_memory.py
import re, sqlite3, datetime as dt, pathlib

# === 個人設定 ===
DB_PATH   = "lianxin_ai.db"
TXT_PATH  = "memory.txt"
USER_ID   = "user_id"   # ← 換成你的 Discord ID
# =================

# ① 以 “【記憶xxx】” 切塊：一塊 = 一條長期記憶
BLOCK_RE = re.compile(r"【記憶\d+】[\s\S]*?(?=【記憶\d+】|\Z)")

# ② 日期正則：2025-2-15、2025/02/15、2025年2月15日 … 都能抓
DATE_RE  = re.compile(r"(\d{4})[./\-\s年](\d{1,2})[./\-\s月](\d{1,2})")

def detect_date(block: str) -> str | None:
    m = DATE_RE.search(block)
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    return f"{y:04d}-{mth:02d}-{d:02d} 00:00:00"

def detect_importance(block: str) -> int:
    if any(k in block for k in ("重大", "轉捩點", "崩潰", "自殺")):
        return 5
    if any(k in block for k in ("社團", "球隊", "家人", "戀愛")):
        return 4
    return 3

# ── 讀檔並切塊 ─────────────────────────
text   = pathlib.Path(TXT_PATH).read_text(encoding="utf-8")
blocks = [blk.strip() for blk in BLOCK_RE.findall(text)]
print(f"🗂️  偵測到 {len(blocks)} 塊待匯入")

# ── 連線 DB，確保欄位存在 ────────────────
with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()

    def ensure_schema():
        for stmt in (
            "ALTER TABLE memories ADD COLUMN created_at TEXT;",
            "ALTER TABLE memories ADD COLUMN importance INTEGER DEFAULT 3;",
        ):
            try:
                cur.execute(stmt)
            except sqlite3.OperationalError:
                pass  # 欄位已存在就跳過

    ensure_schema()

    # ── 寫入 ───────────────────────────
    inserted = 0
    for blk in blocks:
        date_str = detect_date(blk) or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not detect_date(blk):
            print("⚠️  無日期 → 用今天：", blk.splitlines()[0][:40])

        imp = detect_importance(blk)

        cur.execute(
            """INSERT INTO memories
                 (user_id, role, content, created_at, importance)
               VALUES (?,      'memory', ?,       ?,          ?)""",
            (USER_ID, blk, date_str, imp),
        )
        inserted += 1

    conn.commit()

print(f"✅ 匯入完畢：插入 {inserted} 筆")
