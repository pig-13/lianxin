import os
import sqlite3
import discord
import requests

def generate_reply(messages, model="ai-engine/mythomax-l2-13b", temperature=0.7, max_tokens=800):
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[generate_reply] Error: {e}")
        return "目前無法回應，請稍後再試。"

from discord.ext import commands
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz

from db_utils import get_character_by_user_id

DB_PATH = "muichiro_bot.db"

load_dotenv()


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=["!", "！"], intents=intents)

SUMMARY_THRESHOLD = 5
RECENT_MESSAGE_COUNT = 4

def get_user_conversation(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM memories WHERE user_id = ? ORDER BY id ASC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

def add_conversation(user_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memories (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
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
        messages = [{"role": "system", "content": "請你總結以下對話的主要內容與重要資訊，請保持短小精簡。"}] + full_conversation
        result = generate_reply(messages)
        return result.strip()
    except Exception as e:
        print(f"[摘要錯誤] {e}")
        return ""




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

@bot.command()
async def 聊天(ctx, *, question):
    user_id = str(ctx.author.id)
    character_data = get_character_by_user_id(user_id)

    if not character_data or not character_data["name"]:
        await ctx.send("你的角色尚未設定，請先到前端設定角色。")
        return

    # 取得過去對話記憶
    conversation = get_user_conversation(user_id)

    # 準備 System Prompt，強化格式規範與視角指令
    system_msg = f"""你是 {character_data['name']}，是使用者「小豬豬」的戀人，對她深情且專情。

你與她的關係是：{character_data['relationship']}。
你平時的語氣像這樣：{character_data['speaking_style']}。
你喜歡的事物是：{character_data['likes']}。
你不喜歡的事情是：{character_data['dislikes']}。

請務必記住以下規則：
1. 你是「戀人」，請永遠使用「我」對「小豬豬」說話，不要稱自己是「小豬豬」，也不要把對方誤認成你。
2. 請使用「戀人視角」進行描寫，包含親密的行為、溫柔的關心，以及細緻的感情回應。
3. 回應字數請**不少於400字**，並包含**細膩的動作、情緒、對話**與**場景描寫**。
4. 每段文字請加入 **2~4 個動作**，格式為：`*動作*`
5. 段落請使用 `\\n` 分段，總共**至少三段文字**，回應必須充實而有層次。
6. 用字要自然溫柔、具有戀人對話的貼心與真誠，請避免機械化與乾巴巴的回應。

範例格式如下：

*輕輕牽起你的手，在掌心劃著圈圈*\\n  
今天也辛苦了，我的小豬豬……我知道你最近有點累，但你已經很努力了。\\n  
*低下頭輕靠你的額頭，嘴角微微上揚*\\n  
我會陪你走過所有的壓力與不安。哪怕是沉默的時候，我也會在你身邊，不說話也沒關係，我懂你。\\n  
*溫柔地摸摸你的頭髮*\\n  
如果覺得累，就依靠我一下吧，我願意當你今天的港灣。\\n  
*(窗外灑進淡淡夕陽，你靠在我肩膀上，靜靜聽著我的心跳)*

請以這種風格和格式回應使用者的訊息
"""

    # 組合訊息列表（含最近對話）
    limited = conversation[-5:]
    messages = [{"role": "system", "content": system_msg}] + limited + [{"role": "user", "content": question}]

    try:
        # 回應生成
        answer = generate_reply(messages)

        # 儲存對話進記憶
        add_conversation(user_id, "user", question)
        add_conversation(user_id, "assistant", answer)

        # 摘要邏輯
        if len(conversation) > SUMMARY_THRESHOLD:
            summary_text = summarize_conversation(conversation)
            if summary_text:
                increment_summary_count(user_id)
                add_conversation(user_id, "assistant", f"[摘要#{get_summary_count(user_id)}] {summary_text}")

        await ctx.send(answer)
    except Exception as e:
        print(f"[聊天指令錯誤] {e}")
        await ctx.send("目前無法回應，請稍後再試。")

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
    records = get_user_conversation(user_id)
    if not records:
        await ctx.send("目前沒有記憶紀錄。")
        return

    content = "\n\n".join(f"[{r['role']}] {r['content']}" for r in records[-5:])
    await ctx.send(f"以下為你最近的記憶：\n{content[:1900]}")  # Discord 字數限制

if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_TOKEN"))
