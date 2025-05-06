# ────────────────────────────────────────────────────────────────────────
#  Muichiro Discord Bot  ‑  all‑in‑one  (2025‑05‑05)
# ────────────────────────────────────────────────────────────────────────
import os
import sqlite3
import time
from datetime import datetime, timedelta

import asyncio
import pytz
import requests

import discord
from discord.ext import commands, tasks

from dotenv import load_dotenv

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import tiktoken

import aiohttp, json, textwrap, asyncio


# ╭─[ 基本設定 ]──────────────────────────────────────────────────────────╮
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DISCORD_TOKEN      = os.getenv("DISCORD_TOKEN")

DB_PATH            = "muichiro_bot.db"

EMBED_MODEL_NAME   = "all-MiniLM-L6-v2"
EMBED_DIM          = 384
model_embed        = SentenceTransformer(EMBED_MODEL_NAME)

encoding           = tiktoken.get_encoding("cl100k_base")

DAILY_LIMIT        = 1000          # 每日 API 次數
REQUESTS_PER_CHAT  = 2             # 一次聊天 ≈ 2 次 OpenRouter 請求
SUMMARY_THRESHOLD  = 5             # 對話過 N 則後才摘要
RECENT_MESSAGE_COUNT = 3           # 回傳時附帶幾則最近對話
tz                 = pytz.timezone("Asia/Taipei")
# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[ Discord Bot ]───────────────────────────────────────────────────────╮
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=["!", "！"], intents=intents)
# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[ DB：共用工具 ]──────────────────────────────────────────────────────╮
def get_user_conversation(user_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT role, content
            FROM memories
            WHERE user_id = ?
            ORDER BY id ASC
        """, (user_id,))
        rows = cur.fetchall()
    return [{"role": r, "content": c} for r, c in rows]


def add_conversation(user_id: str, role: str, content: str, importance: int = 3):
    ts  = datetime.now(tz).strftime("%F %T")
    emb = model_embed.encode([content])[0].astype(np.float32).tobytes() if role == "memory" else None
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO memories(user_id, role, content, created_at, importance, embedding)
            VALUES(?,?,?,?,?,?)
        """, (user_id, role, content, ts, importance, emb))
        conn.commit()


def clear_conversation(user_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        conn.commit()


def get_summary_count(user_id: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            SELECT COUNT(*) FROM memories
            WHERE user_id=? AND role='memory'
        """, (user_id,))
        row = cur.fetchone()
    return row[0] if row else 0

def get_next_memory_id(user_id: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            SELECT MAX(id) FROM memories
            WHERE user_id = ? AND role = 'memory'
        """, (user_id,))
        row = cur.fetchone()
    return (row[0] or 0) + 1

def get_long_term_memories(user_id: str, limit: int = 50):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT content
            FROM memories
            WHERE user_id=? AND role='memory'
            ORDER BY importance DESC,
                     datetime(COALESCE(created_at,'1970-01-01')) DESC
            LIMIT ?
        """, (user_id, limit))
        rows = cur.fetchall()
    return [r[0] for r in rows]
# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[ API‑log：每日計數 ]───────────────────────────────────────────────────╮
with sqlite3.connect(DB_PATH) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_log (
            day_key TEXT PRIMARY KEY,
            count   INTEGER DEFAULT 0
        )
    """)


def _day_key_now() -> str:
    return datetime.now(tz).strftime("%Y-%m-%d")


def increment_api_counter(amount: int = 1):
    day_key = _day_key_now()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO api_log(day_key, count) VALUES(?, ?)
            ON CONFLICT(day_key) DO UPDATE SET count = count + ?;
        """, (day_key, amount, amount))
        conn.commit()


def get_today_usage() -> int:
    day_key = _day_key_now()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT count FROM api_log WHERE day_key=?", (day_key,))
        row = cur.fetchone()
    return row[0] if row else 0
# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[  記憶向量檢索  ]────────────────────────────────────────────────────╮
def build_index(user_id: str | None = None):
    """回傳 (faiss_index, id_list)。若無記憶則回傳 (None, [])."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, embedding FROM memories
            WHERE embedding IS NOT NULL AND role='memory' {}
        """.format("AND user_id=?" if user_id else ""), (() if user_id is None else (user_id,)))
        rows = cur.fetchall()

    if not rows:
        return None, []

    ids, blobs = zip(*rows)
    vecs = np.vstack([np.frombuffer(b, dtype=np.float32) for b in blobs])
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(vecs)
    return index, list(ids)


def get_similar_memories(user_id: str, query_text: str,
                         top_k: int = 3, max_distance: float | None = 1):
    index, id_map = build_index(user_id)
    if index is None:
        return []

    q = model_embed.encode([query_text])[0].astype(np.float32)
    D, I = index.search(q.reshape(1, -1), min(top_k, index.ntotal))

    pairs = [(id_map[int(pos)], float(dist))
             for pos, dist in zip(I[0], D[0])
             if (max_distance is None or dist < max_distance)]

    if not pairs:
        return []

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT id, content FROM memories WHERE id IN ({','.join('?'*len(pairs))})",
                    [pid for pid, _ in pairs])
        id2content = {i: c for i, c in cur.fetchall()}

    return [(id2content[pid], dist) for pid, dist in pairs if pid in id2content]
# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[ Token 工具 ]────────────────────────────────────────────────────────╮
def estimate_tokens(messages):
    return sum(len(encoding.encode(m["content"])) for m in messages)


def safe_trim(messages, answer_budget=256, max_ctx=8192):
    """超過總 token 時，依序砍最早的非 system 訊息。"""
    while len(messages) > 1 and estimate_tokens(messages) + answer_budget > max_ctx:
        del messages[1]
    return messages
# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[ 生成回覆（含語意檢索） ]─────────────────────────────────────────────╮
# －－完整替換原先的 generate_reply 函式－－
class RateLimitError(RuntimeError):
    """OpenRouter 免費額度已用完：攜帶 reset 時間字串"""
    def __init__(self, msg: str, reset_local: str):
        super().__init__(msg)
        self.reset_local = reset_local


async def generate_reply(
    user_id: str,
    messages: list[dict],
    model: str = "deepseek/deepseek-chat-v3-0324:free",
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> str:
    """呼叫 OpenRouter；遇到免費額度用完時拋 RateLimitError"""

    increment_api_counter(REQUESTS_PER_CHAT)

    # 1) 加入語意記憶
    last_input = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    mems = get_similar_memories(user_id, last_input, top_k=3, max_distance=1)
    if mems:
        mem_txt = "\n".join(f"- {t}" for t, _ in mems)
        messages = [{"role": "system",
                     "content": "以下是過往記憶，可作背景參考，請勿逐句複製：\n" + mem_txt}
                    ] + messages
    # ✅ DEBUG 印出實際撈到的語意記憶
        print("📚 [語意檢索記憶] ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
        for idx, (text, dist) in enumerate(mems, 1):
            print(f"{idx}. 相似度距離={dist:.4f}：{text}")
        print("📚 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    print(f"🧮 總 token 數：約 {estimate_tokens(messages)}")

    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer":  "https://muichiro.local",
        "X-Title":       "Muichiro Bot",
    }

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=45)
    ) as sess:
        for attempt in range(3):
            try:
                async with sess.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload, headers=headers
                ) as r:

                    raw = await r.text()          # 先抓完整回應字串

                    # ── 2) 處理 429：免費額度用完 ────────────────────
                    if r.status == 429:
                        info     = json.loads(raw)
                        err_msg  = info["error"]["message"]

                        # 直接顯示「明天 08:00」為重置點
                        now_local  = datetime.now(tz)
                        tomorrow_8am = (now_local + timedelta(days=1)).replace(
                        hour=8, minute=0, second=0, microsecond=0)
                        reset_str = tomorrow_8am.strftime("%m-%d 08:00")

                        raise RateLimitError(err_msg, reset_str)
                    # ────────────────────────────────────────────────

                    if r.status != 200:
                        print(f"[OpenRouter] HTTP {r.status}\n"
                              f"{textwrap.shorten(raw, 150)}")
                        raise RuntimeError(f"http {r.status}")

                    data = json.loads(raw)
                    if "choices" not in data:
                        print("[OpenRouter] 回傳中無 choices：", raw[:200])
                        raise RuntimeError("missing choices")

                    return data["choices"][0]["message"]["content"].strip()

            except RateLimitError:
                # 往上丟給呼叫端處理（聊天 / 摘要）
                raise
            except Exception as e:
                import traceback
                print(f"[generate_reply] retry {attempt+1}/3 ➜ {repr(e)}")
                traceback.print_exc()  # ★ 顯示完整 traceback
                await asyncio.sleep(5)


    raise RuntimeError("three tries failed")
# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[ 摘要 ]──────────────────────────────────────────────────────────────╮
async def summarize_conversation(user_id, recent_pairs):
    if not recent_pairs:
        return ""

    try:
        sys = {"role": "system", "content": """請將以下對話內容整理成可用於角色記憶系統的摘要，需清楚記錄：
1.對話中的具體人物、事件與時間線
2.角色的心理狀態與變化（如情緒起伏、自我揭露等）
3.發生的重大轉折或決策（如喜歡某人、休學、被罵等）
4.請使用精確扼要的描述方式，不要加入模糊或抽象句子。
5.嚴禁虛構未出現在對話裡的事件或細節，否則請說「我不記得」。"""}

        messages = [sys] + recent_pairs
        budget = 3000
        while estimate_tokens(messages) > budget and len(recent_pairs) > 1:
            recent_pairs.pop(0)
            messages = [sys] + recent_pairs

        print(f"🧠 摘要 token：約 {estimate_tokens(messages)}")
        summary = await generate_reply(user_id, messages)
        print("📦 摘要模型回傳內容：", summary)
        return summary.strip()

    except Exception as e:
        print(f"[摘要錯誤] {e}")
        return ""

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[  Reminders & Tables  ]──────────────────────────────────────────────╮
with sqlite3.connect(DB_PATH) as conn:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS characters(
            user_id TEXT PRIMARY KEY,
            name TEXT, age TEXT, occupation TEXT, relationship TEXT,
            background TEXT, personality TEXT, speaking_style TEXT,
            likes TEXT, dislikes TEXT, extra TEXT
        );

        CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            importance INTEGER DEFAULT 3,
            embedding BLOB
        );

        CREATE TABLE IF NOT EXISTS reminders(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            scheduled TEXT,
            reminder_text TEXT,
            repeat INTEGER DEFAULT 0
        );
    """)
    conn.commit()


@bot.event
async def on_ready():
    print(f"已登入為 {bot.user}")
    check_reminders.start()
# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[ 定時任務：提醒 ]────────────────────────────────────────────────────╮
@tasks.loop(seconds=60)
async def check_reminders():
    now = datetime.now(tz)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, user_id, scheduled, reminder_text, repeat FROM reminders")
        rows = cur.fetchall()

        for rid, uid, sched_str, text, repeat in rows:
            sched_time = datetime.fromisoformat(sched_str)
            if sched_time.tzinfo is None:
                sched_time = sched_time.replace(tzinfo=tz)
            if now >= sched_time:
                user = await bot.fetch_user(int(uid))
                try:
                    await user.send(f"⏰ 提醒你：{text}")
                except Exception as e:
                    print(f"[提醒] 發送給 {uid} 失敗：{e}")

                if repeat:
                    next_time = sched_time + timedelta(days=1)
                    cur.execute("UPDATE reminders SET scheduled=? WHERE id=?",
                                (next_time.isoformat(), rid))
                else:
                    cur.execute("DELETE FROM reminders WHERE id=?", (rid,))
        conn.commit()
# ╰───────────────────────────────────────────────────────────────────────╯

# ╭─[ 指令區塊 ]──────────────────────────────────────────────────────────╮
from db_utils import get_character_by_user_id   # 你原來的 util 保留

@bot.command()
async def 指令(ctx):
    await ctx.send(
        """**📜 可用指令總覽**

🧑‍🎤 角色相關
└ `！查看角色`                 查看自己的角色資料  
└ `！重設角色`                 重置自己的角色資料  
└ `！設定角色 <欄位> <內容>`   設定或更新角色欄位  
   例：！設定角色 性格 溫柔體貼

💬 聊天
└ `！聊天 <訊息>`              與角色聊天（保留對話記憶，含動作）  
   例：！聊天 早安呀～

⏰ 提醒
└ `！提醒 HH:MM <訊息>`        指定「今天」時間一次性提醒  
   例：！提醒 12:00 吃飯  
└ `！提醒 HH:MM 每天 <訊息>`   每天固定時間提醒  
   例：！提醒 21:30 每天 喝水  
└ `！提醒 MM/DD HH:MM <訊息>` 指定日期一次性提醒  
   例：！提醒 05/11 12:00 考試  
└ `！查看已有提醒`             列出目前設定的提醒  
└ `！刪除提醒 <編號>`          刪除指定提醒（先用上條指令查編號）

🧠 記憶管理
└ `！查看記憶`                 查看最近的對話記憶  
└ `！重置記憶`                 清空所有對話記憶  
└ `！刪除記憶 <編號>`          刪除特定記憶片段

🔧 其他工具
└ `！查我ID`                   顯示你的 Discord 使用者 ID  
└ `！查看聊天次數`             檢視剩餘聊天配額"""
    )

# 角色設定 / 查詢 / 重置
# ────────────────────────────────────────────────────────────────────────
@bot.command()
async def 設定角色(ctx, 欄位: str, *, 內容: str):
    """以中文欄位設定角色資訊，例如： ！設定角色 個性 溫柔又呆萌"""
    中文對應 = {
        "名字": "name",
        "年齡": "age",
        "職業": "occupation",
        "關係": "relationship",
        "背景": "background",
        "個性": "personality",
        "說話風格": "speaking_style",
        "喜歡的東西": "likes",
        "不喜歡的東西": "dislikes",
        "補充": "extra"
    }

    if 欄位 not in 中文對應:
        await ctx.send(f"欄位「{欄位}」無效，可用：{', '.join(中文對應.keys())}")
        return

    key = 中文對應[欄位]
    user_id = str(ctx.author.id)

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM characters WHERE user_id=?", (user_id,))
        exists = cur.fetchone()

        if exists:
            cur.execute(f"UPDATE characters SET {key}=? WHERE user_id=?", (內容, user_id))
        else:
            blanks = {v: "" for v in 中文對應.values()}
            blanks[key] = 內容
            cur.execute("""
                INSERT INTO characters
                (user_id, name, age, occupation, relationship, background,
                 personality, speaking_style, likes, dislikes, extra)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (user_id,
                  blanks["name"], blanks["age"], blanks["occupation"], blanks["relationship"],
                  blanks["background"], blanks["personality"], blanks["speaking_style"],
                  blanks["likes"], blanks["dislikes"], blanks["extra"]))
        conn.commit()

    await ctx.send(f"角色欄位「{欄位}」已設定為：{內容}")


@bot.command()
async def 查看角色(ctx):
    user_id = str(ctx.author.id)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM characters WHERE user_id=?", (user_id,))
        char = cur.fetchone()

    if not char or not char[1]:
        await ctx.send("你的角色尚未設定，請使用 `！設定角色` 指令。")
        return

    detail = (
        f"**角色名稱**：{char[1]}\n"
        f"**年齡**：{char[2] or '未設定'}\n"
        f"**職業**：{char[3] or '未設定'}\n"
        f"**角色與我的關係**：{char[4] or '未設定'}\n"
        f"**背景故事**：{char[5] or '未設定'}\n"
        f"**個性**：{char[6] or '未設定'}\n"
        f"**說話風格**：{char[7] or '未設定'}\n"
        f"**喜歡的東西**：{char[8] or '未設定'}\n"
        f"**不喜歡的東西**：{char[9] or '未設定'}\n"
        f"**補充說明**：{char[10] or '無'}"
    )
    await ctx.send(detail)


@bot.command()
async def 重設角色(ctx):
    user_id = str(ctx.author.id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM characters WHERE user_id=?", (user_id,))
        conn.commit()
    await ctx.send("已重置你的角色資料，請重新設定。")

# ────────────────────────────────────────────────────────────────────────
# 記憶 CRUD
# ────────────────────────────────────────────────────────────────────────
@bot.command()
async def 重置記憶(ctx):
    clear_conversation(str(ctx.author.id))
    await ctx.send("已清除你與機器人的所有對話記憶。")


@bot.command()
async def 刪除記憶(ctx, 記憶編號: int):
    user_id = str(ctx.author.id)
    prefix = f"【記憶{記憶編號}】"
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM memories
            WHERE user_id=? AND role='memory' AND content LIKE ?
        """, (user_id, f"{prefix}%"))
        row = cur.fetchone()

        if not row:
            await ctx.send(f"找不到「記憶{記憶編號}」，請確認編號。")
            return

        cur.execute("DELETE FROM memories WHERE id=?", (row[0],))
        conn.commit()
    await ctx.send(f"🗑️ 已刪除記憶（記憶{記憶編號}）。")


@bot.command()
async def 查看記憶(ctx):
    user_id = str(ctx.author.id)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT content FROM memories
            WHERE user_id=? AND role='memory'
            ORDER BY id DESC LIMIT 5
        """, (user_id,))
        rows = cur.fetchall()

    if not rows:
        await ctx.send("目前沒有記憶摘要紀錄。")
        return

    await ctx.send("以下為你最近的記憶摘要：\n" +
                   "\n\n".join(r[0] for r in rows)[:1900])

# ────────────────────────────────────────────────────────────────────────
# 提醒系統
# ────────────────────────────────────────────────────────────────────────
@bot.command()
async def 提醒(ctx, *args):
    """格式：
       ！提醒 HH:MM 訊息
       ！提醒 MM/DD HH:MM 訊息
       在訊息前加「每天 」會變成每日提醒
    """
    if len(args) < 2:
        await ctx.send("❗ 格式錯誤，請使用 `！提醒 HH:MM 訊息` 或 `！提醒 MM/DD HH:MM 訊息`")
        return

    now = datetime.now(tz)
    try:
        # HH:MM
        if ":" in args[0] and "/" not in args[0]:
            hour, minute = map(int, args[0].split(":"))
            sched = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if sched < now:
                sched += timedelta(days=1)
            reminder_text = " ".join(args[1:])

        # MM/DD HH:MM
        elif "/" in args[0] and ":" in args[1]:
            month, day = map(int, args[0].split("/"))
            hour, minute = map(int, args[1].split(":"))
            sched = tz.localize(datetime(now.year, month, day, hour, minute))
            reminder_text = " ".join(args[2:])
        else:
            raise ValueError
    except Exception:
        await ctx.send("❗ 時間格式錯誤，範例：`！提醒 05/11 12:00 考試`")
        return

    # 是否為「每天提醒」
    repeat = 1 if reminder_text.startswith("每天 ") else 0
    if repeat:
        reminder_text = reminder_text[3:].strip()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO reminders(user_id, scheduled, reminder_text, repeat)
            VALUES(?,?,?,?)
        """, (str(ctx.author.id), sched.isoformat(), reminder_text, repeat))
        conn.commit()

    mode = "（每天提醒）" if repeat else "（僅此一次）"
    await ctx.send(f"✅ 已設定：{sched.strftime('%Y-%m-%d %H:%M')} 提醒你：{reminder_text} {mode}")


@bot.command()
async def 查看已有提醒(ctx):
    user_id = str(ctx.author.id)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, scheduled, reminder_text, repeat
            FROM reminders
            WHERE user_id=?
            ORDER BY scheduled ASC
        """, (user_id,))
        rows = cur.fetchall()

    if not rows:
        await ctx.send("你目前沒有任何提醒事項喔～")
        return

    lines = []
    for rid, sched, text, repeat in rows:
        t_str = datetime.fromisoformat(sched).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{rid}. {t_str}（{'每天' if repeat else '一次'}）：{text}")
    await ctx.send("以下是你設定的提醒（用 `！刪除提醒 編號` 刪除）：\n" +
                   "\n".join(lines)[:1900])


@bot.command()
async def 刪除提醒(ctx, reminder_id: int):
    user_id = str(ctx.author.id)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM reminders WHERE id=? AND user_id=?", (reminder_id, user_id))
        row = cur.fetchone()
        if not row:
            await ctx.send(f"找不到編號 {reminder_id} 的提醒。")
            return
        cur.execute("DELETE FROM reminders WHERE id=?", (reminder_id,))
        conn.commit()
    await ctx.send(f"已刪除提醒（編號 {reminder_id}）。")

# ────────────────────────────────────────────────────────────────────────
# 其他輔助指令
# ────────────────────────────────────────────────────────────────────────
@bot.command()
async def 查我ID(ctx):
    await ctx.send(f"你的 Discord user ID 是：`{ctx.author.id}`")

@bot.command(name="查看聊天次數", aliases=["！查看聊天次數"])
async def check_usage(ctx):
    used_req     = get_today_usage()
    used_chat    = used_req // REQUESTS_PER_CHAT
    total_chat   = 454  # ✅ 你明確設定為 454 次聊天
    remain_chat  = max(total_chat - used_chat, 0)
    remain_req   = max(DAILY_LIMIT - used_req, 0)

    await ctx.send(
        f"今天 **早上 08:00** 起：\n"
        f" • 已用 **{used_req}/{DAILY_LIMIT}** 次 API（≈ {used_chat} 次聊天）\n"
        f" • 剩餘 {remain_req} 次（≈ {remain_chat} 次聊天）"
    )

# — 聊天主指令 ------------------------------------------------------------
@bot.command()
async def 聊天(ctx, *, question: str):
    """主聊天指令：自動套用角色、語意記憶，並處理免費額度用完的情況"""

    user_id = str(ctx.author.id)

    # 1) 檢查角色是否已設定
    character_data = get_character_by_user_id(user_id)
    if not character_data or not character_data["name"]:
        await ctx.send("你的角色尚未設定，請先到前端設定角色。")
        return

    # 2) 最近對話（含本輪問題）
    conv   = get_user_conversation(user_id)
    conv.append({"role": "user", "content": question})
    recent = conv[-RECENT_MESSAGE_COUNT:]

    # 3) System Prompt（固定角色指令）
    system_msg = {
        "role": "system",
        "content": (
            f"你是 {character_data['name']}，是使用者的戀人，對她深情且專情。\n"
            f"你與她的關係：{character_data['relationship']}\n"
            f"說話風格：{character_data['speaking_style']}\n"
            f"喜歡：{character_data['likes']}\n"
            f"不喜歡：{character_data['dislikes']}\n\n"
            "請遵守：\n"
            "1. 永遠用「我」對「使用者」說話。\n"
            "2. 加入 *動作*、情緒、場景描寫（戀人視角）。\n"
            "3. 至少 120 字並自然分段。\n"
            "4. 避免冷淡或機械感。\n"
            "5. 請根據背景記憶作答，不得捏造未提及的事件或細節。\n"
        )
    }
    messages = [system_msg] + recent
    messages = safe_trim(messages, answer_budget=256, max_ctx=8192)

    # 4) 呼叫生成（捕捉免費額度已用完）
    try:
        answer = await generate_reply(
            user_id, messages,
            model="deepseek/deepseek-chat-v3-0324:free",
            max_tokens=256
        )
    except RateLimitError as e:
        # 免費額度用完 → 直接告知使用者並結束
        await ctx.send(f"（OpenRouter：{e}；免費額度將在 **{e.reset_local}** 重置）")
        return
    except Exception as e:
        print("[聊天] generate_reply error: ", e)
        await ctx.send("（伺服器忙碌，請稍後再試…）")
        return

    # 5) 寫入對話記憶
    add_conversation(user_id, "user",      question, importance=3)
    add_conversation(user_id, "assistant", answer,   importance=3)

    # 6) 嘗試摘要（摘要 hit limit 時直接跳過）
    used_req = get_today_usage()
    used_chat = used_req // REQUESTS_PER_CHAT

    if used_chat > 0 and used_chat % 5 == 0:
        try:
            # 最近 5 輪對話（每輪包含 user + assistant）
            recent_pairs = []
            for m in reversed(conv):
                if m["role"] in ["user", "assistant"]:
                    recent_pairs.insert(0, m)
                if len(recent_pairs) >= 10:
                    break

            summary = await summarize_conversation(user_id, recent_pairs)
            if summary:
                sid = get_next_memory_id(user_id)
                today = datetime.now(tz).strftime("%Y-%m-%d")
                content = f"【記憶{sid}】{today} {summary}"
                add_conversation(user_id, "memory", content, importance=4)
                await ctx.send(f"🧠 已新增記憶：記憶{sid}")


        except RateLimitError:
            # 摘要也吃到免費額度限制就不做摘要，避免洗版
            pass
        except Exception as e:
            print("[聊天] summarize error: ", e)
    # 7) 傳送回覆
    await ctx.send(answer)
# ╰───────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
