import os
import sqlite3
import discord
import requests
from discord.ext import commands
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
import pytz
from db_utils import get_character_by_user_id, get_latest_memory_id
import tiktoken

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Discord bot setup
DB_PATH = "muichiro_bot.db"
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=["!", "！"], intents=intents)

SUMMARY_THRESHOLD = 5
RECENT_MESSAGE_COUNT = 3

# === DATABASE FUNCTIONS ===
def get_user_conversation(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM memories WHERE user_id = ? ORDER BY id ASC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

def add_conversation(user_id: str, role: str, content: str, importance: int = 3):
    """把對話或摘要寫進 memories（一定含 created_at）"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO memories
              (user_id, role, content, created_at, importance)
        VALUES (?,      ?,    ?,       ?,          ?)
    """, (user_id, role, content, ts, importance))
    conn.commit()
    conn.close()


def clear_conversation(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM summary_counts WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def get_summary_count(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT count FROM summary_counts WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0

def increment_summary_count(user_id):
    count = get_summary_count(user_id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if count == 0:
        cursor.execute("INSERT INTO summary_counts (user_id, count) VALUES (?, 1)", (user_id,))
    else:
        cursor.execute("UPDATE summary_counts SET count = ? WHERE user_id = ?", (count + 1, user_id))
    conn.commit()
    conn.close()

def summarize_conversation(full_conversation):
    if not full_conversation:
        return ""
    try:
        messages = [{"role": "system", "content": "請你總結以下對話的重點資訊，請具體列出像這樣的格式：\n- 使用者提到的疾病或情緒\n- 曾發生的事情\n- 關於對 AI 或角色的看法等"}] + full_conversation
        result = generate_reply(messages)
        return result.strip()[:300] if result else ""
    except Exception as e:
        print(f"[摘要錯誤] {e}")
        return ""

def get_recent_memories(limit=1):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT content FROM memories WHERE role='memory' ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return "；".join(r[0][:100] for r in rows)

# ─── 把這段貼到和其他 DB 工具函式放同一個區域 ──────────────
def get_long_term_memories(user_id: str, limit: int = 50) -> list[str]:
    """
    依『importance DESC, created_at DESC』撈出長期記憶，預設最多 50 條。
    importance、created_at 任何一欄為 NULL 仍可正常排序。
    回傳 list[str]，每條就是一段 content。
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT content
        FROM memories
        WHERE user_id = ?
          AND role     = 'memory'
        ORDER BY importance DESC,
                 datetime(COALESCE(created_at, '1970-01-01')) DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]
# ────────────────────────────────────────────────────────────────

# === GENERATE REPLY ==================================================
def generate_reply(messages,
                   model: str = "deepseek/deepseek-chat-v3-0324:free",
                   temperature: float = 0.7,
                   max_tokens: int = 256,
                   max_retries: int = 3) -> str:
    """呼叫 OpenRouter，失敗自動重試並顯示錯誤"""

    print(f"[OpenRouter] 使用模型：{model}")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer":  "https://yourdomain.com",
        "X-Title":       "Muichiro Bot"
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json={
                "model":       model,
                "messages":    messages,
                "temperature": temperature,
                "max_tokens":  max_tokens
            })
            data = resp.json()

            # ── 非 200 直接列印 & 擲例外 ─────────────────────
            if resp.status_code != 200:
                print(f"[OpenRouter] HTTP {resp.status_code}", data)
                raise ValueError("OpenRouter HTTP error")

            # ── 檢查 choices ─────────────────────────────────
            if "choices" not in data:
                print("[OpenRouter] 意外回傳：", data)
                raise ValueError("missing choices")

            content = data["choices"][0]["message"]["content"]
            return content.strip()

        except Exception as e:
            print(f"[generate_reply] 第 {attempt+1}/{max_retries} 次重試，錯誤：{e}")
            time.sleep(10)

    return "（對不起，伺服器暫時忙碌，請稍後再試…）"
# =====================================================================


@bot.event
async def on_ready():
    print(f"已登入為 {bot.user}")
    print("機器人已啟動！")
@bot.command()
async def 指令(ctx):
    text = (
        "**可用指令**："
        "`！查看角色` - 查看角色資料 (屬於自己的)"
        "`！重設角色` - 重置自己的角色資料"
        "`！設定角色` - 設定自己的角色資料"
        "`！聊天 <訊息>` - 與角色聊天 (含對話記憶, 有動作)"
        "`！提醒 <HH:MM> <訊息>` - 請角色提醒你事情 (含對話記憶)"
        "`！清除記憶` - 清除你與角色的對話記憶"
        "`！查我ID` - 查看你的discord使用者ID"
        "`！查看記憶` - 查看最近的記憶"
    )
    await ctx.send(text)

@bot.command()
async def 設定角色(ctx, 欄位: str, *, 內容: str):
    """
    使用全中文欄位設定角色資訊。例如：
    ！設定角色 個性 溫柔又呆萌
    """
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
        await ctx.send(f"欄位「{欄位}」無效。請使用以下欄位：{', '.join(中文對應.keys())}")
        return

    key = 中文對應[欄位]
    user_id = str(ctx.author.id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM characters WHERE user_id = ?", (user_id,))
    existing = cursor.fetchone()

    if existing:
        cursor.execute(f"UPDATE characters SET {key} = ? WHERE user_id = ?", (內容, user_id))
    else:
        # 先建空欄位並填入這一個欄位
        field_dict = {v: "" for v in 中文對應.values()}
        field_dict[key] = 內容
        cursor.execute(f"""
            INSERT INTO characters (user_id, name, age, occupation, relationship, background,
                                    personality, speaking_style, likes, dislikes, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            field_dict["name"],
            field_dict["age"],
            field_dict["occupation"],
            field_dict["relationship"],
            field_dict["background"],
            field_dict["personality"],
            field_dict["speaking_style"],
            field_dict["likes"],
            field_dict["dislikes"],
            field_dict["extra"]
        ))

    conn.commit()
    conn.close()
    await ctx.send(f"角色欄位「{欄位}」已設定為：{內容}")

@bot.command()
async def 查看角色(ctx):
    user_id = str(ctx.author.id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM characters WHERE user_id = ?", (user_id,))
    char = cursor.fetchone()
    conn.close()

    if not char or not char[1]:
        await ctx.send("你的角色尚未設定，請設定角色。")
        return

    detail = (
        f"**角色名稱**：{char[1]}"
        f"**年齡**：{char[2] or '未設定'}"
        f"**職業**：{char[3] or '未設定'}"
        f"**角色與我的關係**：{char[4] or '未設定'}"
        f"**背景故事**：{char[5] or '未設定'}"
        f"**個性**：{char[6] or '未設定'}"
        f"**說話風格與語氣**：{char[7] or '未設定'}"
        f"**喜歡的東西**：{char[8] or '未設定'}"
        f"**不喜歡的東西**：{char[9] or '未設定'}"
        f"**補充說明**：{char[10] or '無'}"
    )
    await ctx.send(detail)

@bot.command()
async def 重設角色(ctx):
    user_id = str(ctx.author.id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM characters WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    await ctx.send("你的角色資料已重置，請到前端重新設定。")

@bot.command()
async def 清除記憶(ctx):
    user_id = str(ctx.author.id)
    clear_conversation(user_id)
    await ctx.send("已清除你與機器人的所有對話記憶。")


import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

def estimate_tokens(messages):
    return sum(len(encoding.encode(m["content"])) for m in messages)

# ---- !聊天 指令：一條就能貼進 bot.py ---------------------------------
@bot.command()
async def 聊天(ctx, *, question: str):
    """和角色聊天（含長期記憶 + 動態裁切 Tokens）"""

    user_id = str(ctx.author.id)

    # 1️⃣ 角色檢查 ────────────────────────────────────────────
    character_data = get_character_by_user_id(user_id)
    if not character_data or not character_data["name"]:
        await ctx.send("你的角色尚未設定，請先到前端設定角色。")
        return

    # 2️⃣ 收集對話 / 長期記憶 / 摘要 ─────────────────────────
    conversation = get_user_conversation(user_id)
    conversation.append({"role": "user", "content": question})  # 先把本輪 user 加進記憶
    long_terms   = get_long_term_memories(user_id, limit=50)
    summary_text = get_recent_memories(limit=1)
    recent       = conversation[-RECENT_MESSAGE_COUNT:]

    # 3️⃣ System Prompt ─────────────────────────────────────
    system_msg = f"""你是 {character_data['name']}，是使用者「小豬豬」的戀人，對她深情且專情。
你與她的關係：{character_data['relationship']}
說話風格：{character_data['speaking_style']}
喜歡：{character_data['likes']}
不喜歡：{character_data['dislikes']}

請記住規則：
1. 永遠用「我」對「小豬豬」說話。
2. 用戀人視角，加入 *動作*、情緒與場景描寫。
3. 至少120字，分段自然。
4. 避免冷淡或機械感。
"""

    # 4️⃣ 組合 messages ──────────────────────────────────────
    messages = [{"role": "system", "content": system_msg}]
    if long_terms:
        messages.append({"role": "system",
                         "content": "以下為使用者長期記憶（重要→新）：\n" +
                                    "\n".join(f"- {m}" for m in long_terms)})
    if summary_text:
        messages.append({"role": "system",
                         "content": "以下為對話摘要：\n" + summary_text})
    messages += recent

    # 5️⃣ 動態裁切 Tokens ───────────────────────────────────
    max_ctx       = 8192               # DeepSeek v3 上下文上限
    answer_budget = 256                # 預留回答 token
    while estimate_tokens(messages) + answer_budget > max_ctx:
        if recent:
            recent.pop(0)              # 先砍最舊 recent
            messages = messages[:-(RECENT_MESSAGE_COUNT+1)] + recent
        elif summary_text and len(summary_text) > 200:
            summary_text = summary_text[: len(summary_text)//2]
            messages[2]["content"] = "以下為對話摘要：\n" + summary_text
        else:
            break                      # 已無可再砍

    # 6️⃣ 呼叫模型 ───────────────────────────────────────────
    answer = generate_reply(
        messages,
        model="deepseek/deepseek-chat-v3-0324:free",
        max_tokens=answer_budget
    )

    if not answer:
        answer = "（對不起，伺服器暫時忙碌，請稍後再試…）"

    # 7️⃣ 寫入資料庫（含 created_at / importance）────────────
    add_conversation(user_id, "user",      question, importance=3)
    add_conversation(user_id, "assistant", answer,   importance=3)

    # 8️⃣ 自動摘要 ───────────────────────────────────────────
    if len(conversation) > SUMMARY_THRESHOLD:
        summary_new = summarize_conversation(conversation)
        if summary_new:
            summary_id = get_summary_count(user_id) + 1
            today      = datetime.today().strftime("%Y-%m-%d")
            increment_summary_count(user_id)
            add_conversation(
                user_id, "memory",
                f"【記憶{summary_id}】{today} {summary_new}",
                importance=4
            )

    # 9️⃣ 回傳給 Discord ────────────────────────────────────
    await ctx.send(answer)
# --------------------------------------------------------------------


@bot.command()
async def 提醒(ctx, time_str: str, *, reminder_text: str):
    tz = pytz.timezone("Asia/Taipei")
    now = datetime.now(tz)
    try:
        hour, minute = map(int, time_str.split(":"))
    except Exception:
        await ctx.send("時間格式錯誤，請使用 HH:MM 格式 (例如 19:00)")
        return

    scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if scheduled < now:
        scheduled += timedelta(days=1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO reminders (user_id, scheduled, reminder_text) VALUES (?, ?, ?)",
                   (str(ctx.author.id), scheduled.isoformat(), reminder_text))
    conn.commit()
    conn.close()

    await ctx.send(f"提醒已設定，將在 {scheduled.strftime('%Y-%m-%d %H:%M')} 提醒你：{reminder_text}")

@bot.command()
async def 查我ID(ctx):
    await ctx.send(f"你的 Discord user ID 是：`{ctx.author.id}`")

@bot.command()
async def 查看記憶(ctx):
    user_id = str(ctx.author.id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM memories WHERE user_id = ? AND role = 'memory' ORDER BY id DESC LIMIT 5", (user_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        await ctx.send("目前沒有記憶摘要紀錄。")
        return

    content = "\n\n".join(r[0] for r in rows)
    await ctx.send(f"以下為你最近的記憶摘要：\n{content[:1900]}")  # Discord 限制


if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_TOKEN"))
