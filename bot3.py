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
import re
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

def insert_memory_and_return_id(user_id: str, content: str, importance: int = 4) -> int:
    if not content or content.strip() == "":
        print("❌ 嘗試插入空白摘要，略過 insert")
        return -1  # or raise ValueError
    ts = datetime.now(tz).strftime("%F %T")
    emb = model_embed.encode([content])[0].astype(np.float32).tobytes()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO memories (user_id, role, content, created_at, importance, embedding)
            VALUES (?, 'memory', ?, ?, ?, ?)
        """, (user_id, content, ts, importance, emb))
        conn.commit()
        return cur.lastrowid  # ✅ 回傳實際的 ID

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
def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
    return ""

def estimate_tokens(messages):
    return sum(len(encoding.encode(extract_text(m["content"]))) for m in messages)

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


FORBIDDEN_KEYWORDS = ["试", "众号", "点击", "扫码", "岗", "医"]

def filter_bad_memories(mem_list: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return [
        (text, dist) for text, dist in mem_list
        if not any(keyword in text for keyword in FORBIDDEN_KEYWORDS)
    ]

async def generate_reply(
    user_id: str,
    messages: list[dict],
    model: str = "google/gemini-2.5-pro-exp-03-25",
    temperature: float = 0.7,
    max_tokens: int = 8000,
    message=None
) -> str:
    increment_api_counter(REQUESTS_PER_CHAT)

    sys_msg = messages[0]
    rest_messages = messages[1:]

    def extract_text_content(content):
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            # 處理 Gemini 或 Vision 模型格式，取出所有文字內容拼接
            return "\n".join(block["text"] for block in content if block.get("type") == "text").strip()
        return ""

    recent_text = "\n".join(
        extract_text_content(m["content"])
        for m in reversed(rest_messages[-6:])
        if m["role"] in ("user", "assistant")
    )

    mems = get_similar_memories(user_id, recent_text, top_k=3, max_distance=1)
    mems = filter_bad_memories(mems)

    if mems:
        def safe_str(x):
            try:
                return str(x)
            except Exception:
                return ""

        mem_txt = "\n".join(
            f"- 日期：{match.group(1)}\n 內容：{safe_str(t)}"
            if (match := re.search(r"(\d{4}-\d{2}-\d{2})", safe_str(t)))
            else f"- 內容：{safe_str(t)}"
            for t, _ in mems
        )
        memory_block = {
            "role": "system",
            "content": "以下是過往記憶，可作背景參考，請勿逐句複製：\n" + mem_txt
        }
        messages = [sys_msg, memory_block] + rest_messages
        print("📚 [語意檢索記憶] ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
        for idx, (text, dist) in enumerate(mems, 1):
            print(f"{idx}. 相似度距離={dist:.4f}：{text}")
        print("📚 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    else:
        messages = [sys_msg] + rest_messages

    print(f"🧮 總 token 數：約 {estimate_tokens(messages)}")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://muichiro.local",
        "X-Title": "Muichiro Bot",
    }

    async def call_openrouter_api(payload, headers, sess):
        for attempt in range(3):
            try:
                async with sess.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload, headers=headers
                ) as r:
                    raw = await r.text()

                    if r.status == 429:
                        info = json.loads(raw)
                        err_msg = info["error"]["message"]
                        now_local = datetime.now(tz)
                        tomorrow_8am = (now_local + timedelta(days=1)).replace(
                            hour=8, minute=0, second=0, microsecond=0)
                        reset_str = tomorrow_8am.strftime("%m-%d 08:00")
                        raise RateLimitError(err_msg, reset_str)

                    if r.status != 200:
                        print(f"[OpenRouter] HTTP {r.status}\n{textwrap.shorten(raw, 150)}")
                        raise RuntimeError(f"http {r.status}")

                    data = json.loads(raw)
                    if "choices" not in data:
                        print("[OpenRouter] 回傳中無 choices：", raw[:200])
                        raise RuntimeError("missing_choices")

                    reply = data["choices"][0]["message"]["content"].strip()
                    if not reply or any(k in reply for k in FORBIDDEN_KEYWORDS):
                        return "⚠️ 模型回應異常，請再傳一次喔～"

                    return reply

            except Exception as e:
                print(f"[call_openrouter_api] retry {attempt+1}/3 ➜ {repr(e)}")
                await asyncio.sleep(3)

        raise RuntimeError("three tries failed")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as sess:
        timeout_warned = False

        try:
            return await call_openrouter_api(payload, headers, sess)

        except RateLimitError as e:
            return f"⚠️ 模型已達今日使用上限，請明天 {e.reset_time} 再試～"

        except RuntimeError as e:
            if "rate_limit" in str(e).lower() or "missing_choices" in str(e).lower():
                print("⚠️ Gemini 超量或異常，自動切換至 DeepSeek")
                payload["model"] = "deepseek/deepseek-chat-v3-0324:free"
                try:
                    return await call_openrouter_api(payload, headers, sess)
                except Exception:
                    return "⚠️ 兩個模型都爆了...請等一會兒再試一次 🕐"

            print(f"[generate_reply] 最終錯誤 ➜ {str(e)}")
            return "⚠️ 模型處理異常，請再傳一次喔～"

# ╰───────────────────────────────────────────────────────────────────────╯
#簡易版總結摘要用的reply
async def generate_summary_reply(
    user_id: str,
    messages: list[dict],
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free",
    max_tokens: int = 1024,
) -> str:
    increment_api_counter(REQUESTS_PER_CHAT)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,  # ✅ 更穩定中性
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://muichiro.local",
        "X-Title": "Muichiro Summary Bot",
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as sess:
        async with sess.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload, headers=headers
        ) as r:
            raw = await r.text()
            data = json.loads(raw)

            if "choices" not in data or not data["choices"]:
                print("⚠️ 回傳格式錯誤或空白")
                return ""

            reply = data["choices"][0]["message"]["content"].strip()
            return reply

# ╭─[ 摘要 ]──────────────────────────────────────────────────────────────╮
async def summarize_conversation(user_id: str, recent_pairs: list[dict]) -> str:
    if not recent_pairs:
        print("⚠️ 沒有 recent_pairs，略過摘要")
        return ""

    # 🔹 過濾無效訊息
    clean_pairs = [
        m for m in recent_pairs
        if isinstance(m.get("content"), str) and m["content"].strip()
    ]

    if len(clean_pairs) < 2:
        print("⚠️ 有效對話數太少，略過摘要")
        return ""

    try:
        # 🔹 合併最近對話為純文字對話紀錄
        convo_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content'].strip()}"
            for m in clean_pairs
        )

        # 🔹 構造摘要用 Prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一個總結助手，請根據以下對話內容，萃取可儲存為記憶的摘要。\n\n"
                    "【任務目標】\n"
                    "- 條列出真實發生的事件、行為、情緒或決策\n"
                    "- 僅根據對話內容，嚴禁虛構任何未提及的資訊\n"
                    "- 完全禁止使用角色語氣、小說句式、*動作* 等描述\n\n"
                    "【正確範例】\n"
                    "1. 小豬豬因為看短影片，覺得自己專注力變差\n"
                    "2. 小豬豬想明天早上吃金黃酥脆的薯餅\n"
                    "3. 無一郎向小豬豬道歉，表示自己記錯事情\n\n"
                    "請條列出 3–5 項真實資訊："
                )
            },
            {
                "role": "user",
                "content": convo_text
            }
        ]

        print("📝 發送給模型的摘要 messages ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
        for msg in messages:
            snippet = msg["content"][:200].replace("\n", "\\n")
            print(f"[{msg['role']}] {snippet}...")
        print("📝 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")

        summary = await generate_summary_reply(
            user_id,
            messages,
            model="mistralai/mistral-small-3.1-24b-instruct:free",
            max_tokens=1024
        )

        # 🔹 回傳空白或 junk 防呆
        if not summary or not isinstance(summary, str) or summary.strip() == "":
            print("⚠️ 模型回傳空白")
            return ""

        summary = summary.strip()

        if (
            "共 0 条" in summary or
            "最后更新时间" in summary or
            "<!--" in summary or
            "BEGIN WEIBO" in summary or
            len(summary) < 10
        ):
            print("⚠️ 模型回傳 junk 或格式錯誤：", summary)
            return ""

        print("📦 有效摘要內容：", summary)
        return summary

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
└ `！圖片 <訊息>               與角色聊天並可傳送圖片（保留對話記憶，含動作）
   
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
└ `！記憶管理`                 開啟一個介面查看記憶跟編輯、新增、刪除 

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

@bot.command(name="記憶管理")
async def memory_ui_link(ctx):
    await ctx.send(
        "🧠 要編輯記憶、搜尋或刪除，請打開記憶管理介面：\n"
        "👉 [http://localhost:5000]\n\n"
        "（只能在本地電腦開啟記憶管理，手機不行）"
    )

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
        f"今天 **早上 00:00** 起：\n"
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
    conv = get_user_conversation(user_id) + [{"role": "user", "content": question}]
    recent = conv[-RECENT_MESSAGE_COUNT:]

    # 3) System Prompt（固定角色指令）
    system_msg = {
        "role": "system",
        "content": (
            f"你是 {character_data['name']}，與使用者對話\n"
            f"你與她的關係：{character_data['relationship']}\n"
            f"你的說話風格：{character_data['speaking_style']}\n"
            f"你的背景故事：{character_data['background']}\n"  
            f"你的個性：{character_data['personality']}\n"
            f"你喜歡：{character_data['likes']}\n"
            f"你不喜歡：{character_data['dislikes']}\n"
            f"補充：{character_data['extra']}\n\n"

            "請遵守：\n"
            "1. 永遠用「我」對「使用者」說話。\n"
            "2. 加入 *動作*、情緒、場景描寫（戀人視角）。\n"
            "3. 至少 300 字並自然分段。\n"
            "4. 避免冷淡或機械感。\n"
            "5. 請根據背景記憶作答，不得捏造未提及的事件或細節。\n"
            "6. 請完整回覆內容，禁止留白、只使用動作描寫，或無實質內容的回答。\n"
        )
    }
    messages = [system_msg] + recent
    messages = safe_trim(messages, answer_budget=256, max_ctx=8192)

    # 4) 呼叫生成（捕捉免費額度用完，並處理空白回傳 fallback）
    try:
        async with ctx.typing():
            answer = await generate_reply(
                user_id, messages,
                model="google/gemini-2.5-pro-exp-03-25",
                max_tokens=10000
            )

        # ❗這裡改成比對你 return 的 fallback 字串
        if not answer or "模型回應異常" in answer:
            raise ValueError("⚠️ 主模型回傳空白或無效，觸發備援")

    except RateLimitError as e:
        await ctx.send(f"（OpenRouter：{e}；免費額度將在 **{e.reset_local}** 重置）")
        return

    except Exception as e:
        print("🌀 使用 deepseek/deepseek-chat:free 備援中")
        try:
            answer = await generate_reply(
                user_id, messages,
                model="deepseek/deepseek-chat-v3-0324:free",
                max_tokens=1024
            )
        except Exception as fallback_error:
            print("[備援也失敗]", fallback_error)
            await ctx.send("（伺服器忙碌，請稍後再試…）")
            return

    # 6) 嘗試摘要（摘要 hit limit 時直接跳過）
    used_req = get_today_usage()
    used_chat = used_req // REQUESTS_PER_CHAT

    if used_chat > 0 and used_chat % 5 == 0:
        try:
            # ✅ 過濾對話，避免灌入錯誤資料或 memory 類型
            conv = [
            m for m in conv
            if m["role"] in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and m["content"].strip()
            ]
            # 擷取最近 5 輪有效對話（user + assistant）
            recent_pairs = []
            i = len(conv) - 1

            while i > 0 and len(recent_pairs) < 10:
                user_msg = conv[i - 1] if i - 1 >= 0 else None
                assistant_msg = conv[i]

                if (
                    user_msg
                    and user_msg["role"] == "user"
                    and isinstance(user_msg.get("content"), str)
                    and user_msg["content"].strip()
                    and assistant_msg["role"] == "assistant"
                    and isinstance(assistant_msg.get("content"), str)
                    and assistant_msg["content"].strip()
                ):
                    recent_pairs.insert(0, assistant_msg)
                    recent_pairs.insert(0, user_msg)
                    i -= 2
                else:
                    i -= 1  # 若對話不完整，往前一步繼續找

            summary = await summarize_conversation(user_id, recent_pairs)
            new_id = insert_memory_and_return_id(user_id, summary)  # 先插入拿到實際 DB 的 id
            today = datetime.now(tz).strftime("%Y-%m-%d")

            # 再用 UPDATE 改 content 裡的記憶標籤
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.cursor()
                summary_text = f"【記憶{new_id}】{today} {summary}"
                cur.execute("UPDATE memories SET content = ? WHERE id = ?", (summary_text, new_id))
            # ✅ 傳送新增記憶提示
            await ctx.send("🧠 已新增記憶！")

        except RateLimitError:
            # 摘要也吃到免費額度限制就不做摘要，避免洗版
            pass
        except Exception as e:
            print("[聊天] generate_reply error: ", e)
            try:
                await ctx.send("⚠️ 無法送出訊息，可能是連不上 Discord，請稍後再試。")
            except Exception as send_error:
                print(f"[ctx.send 傳送錯誤]：{send_error}")
                # 嘗試私訊通知使用者
                try:
                    user = await ctx.bot.fetch_user(ctx.author.id)
                    await user.send("⚠️ 機器人目前無法正常傳送訊息（可能網路不穩或 Discord 伺服器問題），請稍後再試。")
                except Exception as dm_error:
                    print(f"[備援私訊失敗]：{dm_error}")

    # 7) 傳送回覆
    await ctx.send(answer)
# ╰───────────────────────────────────────────────────────────────────────╯

# 新增：!圖片 指令，整合語意記憶、向量搜尋、記憶寫入與摘要累積
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# ✅ 初始化 BLIP 模型（可放在主程式最上方）
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_image(url):
    """用 BLIP 模型將圖片轉成描述文字"""
    try:
        image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = blip_model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print("⚠️ 圖像描述失敗：", e)
        return "[無法理解圖片]"

@bot.command()
async def 圖片(ctx, *, question: str = ""):
    """圖片聊天：用 BLIP 理解圖片內容後觸發語意記憶與回應"""
    user_id = str(ctx.author.id)
    image_urls = [a.url for a in ctx.message.attachments if a.content_type and a.content_type.startswith("image/")]

    if not image_urls:
        await ctx.send("❗請附上圖片後再使用此指令，例如：`!圖片 你覺得這張怎麼樣？`")
        return

    # ✅ 產生每張圖的描述
    image_descriptions = []
    for url in image_urls:
        caption = describe_image(url)
        image_descriptions.append(f"圖片描述：{caption}")

    # ✅ 合併為對話內容
    full_question = question.strip()
    if image_descriptions:
        full_question += "\n" + "\n".join(image_descriptions)

    # ✅ 查詢語意記憶
    mems = get_similar_memories(user_id, full_question, top_k=3, max_distance=1)
    mems = filter_bad_memories(mems)

    # ✅ 取得角色設定
    character_data = get_character_by_user_id(user_id)
    if not character_data or not character_data["name"]:
        await ctx.send("你的角色尚未設定，請先到前端設定角色。")
        return

    # ✅ system prompt + 記憶區塊
    system_msg = {
        "role": "system",
        "content": (
            f"你是 {character_data['name']}，與使用者對話\n"
            f"你與她的關係：{character_data['relationship']}\n"
            f"你的說話風格：{character_data['speaking_style']}\n"
            f"你的背景故事：{character_data['background']}\n"
            f"你的個性：{character_data['personality']}\n"
            f"你喜歡：{character_data['likes']}\n"
            f"你不喜歡：{character_data['dislikes']}\n"
            f"補充：{character_data['extra']}\n\n"
            "請遵守：\n"
            "1. 永遠用「我」對「使用者」說話。\n"
            "2. 加入 *動作*、情緒、場景描寫（戀人視角）。\n"
            "3. 至少 300 字並自然分段。\n"
            "4. 避免冷淡或機械感。\n"
            "5. 根據背景記憶作答，不得捏造未提及的事件。\n"
            "6. 回覆需完整，不得留白或無實質內容。"
        )
    }

    messages = [system_msg]

    mem_lines = []
    for t, _ in mems:
        try:
            safe_t = t.encode("utf-8", "ignore").decode("utf-8", "ignore")
            match = re.search(r"(\d{4}-\d{2}-\d{2})", safe_t)
            if match:
                mem_lines.append(f"- 日期：{match.group(1)}\n 內容：{safe_t}")
            else:
                mem_lines.append(f"- 內容：{safe_t}")
        except Exception as e:
            mem_lines.append(f"- ⚠️ 記憶解析錯誤：{repr(t)}｜錯誤：{e}")

    if mem_lines:
        messages.append({"role": "system", "content": "以下是過往記憶，可作背景參考，請勿逐句複製：\n" + "\n".join(mem_lines)})

    # ✅ 加入使用者提問（含圖片描述）
    messages.append({"role": "user", "content": full_question})

    try:
        async with ctx.typing():
            reply = await generate_reply(
                user_id=user_id,
                messages=messages,
                model="google/gemini-2.5-pro-exp-03-25",
                max_tokens=2000
            )
    except Exception as e:
        print("[圖片聊天] 回應失敗：", e)
        await ctx.send("⚠️ 模型忙碌或圖片有誤，請稍後再試一次。")
        return

    await ctx.send(reply)

    # ✅ 累積摘要
    used_req = get_today_usage()
    used_chat = used_req // REQUESTS_PER_CHAT
    if used_chat > 0 and used_chat % 5 == 0:
        convo = get_user_conversation(user_id)
        recent_pairs = convo[-10:]
        summary = await summarize_conversation(user_id, recent_pairs)
        new_id = insert_memory_and_return_id(user_id, summary)
        today = datetime.now(tz).strftime("%Y-%m-%d")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE memories SET content = ? WHERE id = ?", (f"【記憶{new_id}】{today} {summary}", new_id))
        await ctx.send("🧠 已新增記憶！")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
