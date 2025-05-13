import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading
import os
import sys

# ========== 🧠 防止子程序重複啟動主視窗 ==========
# 如果收到 --child 參數，就不開 UI
if "--child" in sys.argv:
    sys.exit(0)
# 如果是打包後執行的，就切換到 PyInstaller 的臨時目錄
if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)


# ========== 📦 若未偵測到 .env 檔案，顯示設定表單 ==========
def show_env_form():
    def save_and_close():
        with open(".env", "w", encoding="utf-8") as f:
            f.write(f"DISCORD_TOKEN={discord_var.get()}\n")
            f.write(f"OPENROUTER_API_KEY={api_key_var.get()}\n")
            f.write(f"USER_ID={user_id_var.get()}\n")
            f.write(f"ORDER_CODE={order_code_var.get()}\n")
        form.destroy()

    form = tk.Tk()
    form.title("戀芯 AI 安裝設定")

    discord_var = tk.StringVar()
    api_key_var = tk.StringVar()
    user_id_var = tk.StringVar()
    order_code_var = tk.StringVar()

    tk.Label(form, text="DISCORD_TOKEN").grid(row=0, column=0, sticky="w")
    tk.Entry(form, textvariable=discord_var, width=50).grid(row=0, column=1)

    tk.Label(form, text="OPENROUTER_API_KEY").grid(row=1, column=0, sticky="w")
    tk.Entry(form, textvariable=api_key_var, width=50).grid(row=1, column=1)

    tk.Label(form, text="Discord 使用者 ID").grid(row=2, column=0, sticky="w")
    tk.Entry(form, textvariable=user_id_var, width=50).grid(row=2, column=1)

    tk.Label(form, text="訂單序號").grid(row=3, column=0, sticky="w")
    tk.Entry(form, textvariable=order_code_var, width=50).grid(row=3, column=1)

    tk.Button(form, text="儲存並開始", command=save_and_close).grid(row=4, columnspan=2, pady=10)
    form.mainloop()

# 只有在打包後執行，且沒有 .env 檔案時才顯示設定表單
if getattr(sys, 'frozen', False) and not os.path.exists(".env"):
    show_env_form()

# ========== 🧠 全域子程序 ==========
bot_process = None
flask_process = None

# ========== 🌐 啟動記憶頁面 ==========
def launch_memory_ui():
    global flask_process
    flask_process = subprocess.Popen(
        [sys.executable, "memory_ui.py", "--child"],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

# ========== 🤖 啟動機器人 ==========
def start_bot():
    print("🧪 啟動 bot.py 的路徑是：", os.getcwd())
    print("🧪 呼叫指令：", [sys.executable, "bot.py", "--child"])
    global bot_process
    if bot_process is None:
        log_output.insert(tk.END, "✅ 啟動中...\n")
        bot_process = subprocess.Popen(
            [sys.executable, os.path.join(os.getcwd(), "bot.py"), "--child"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        log_output.insert(tk.END, "✅ 已啟動戀芯 AI 機器人\n請到 http://localhost:5000 查看記憶介面\n\n")
        threading.Thread(target=read_output, daemon=True).start()

# ========== 🛑 關閉機器人 ==========
def stop_bot():
    global bot_process
    if bot_process:
        bot_process.terminate()
        log_output.insert(tk.END, "🛑 已關閉戀芯 AI 機器人\n")
        bot_process = None

# ========== 🔁 重新啟動 ==========
def restart_bot():
    stop_bot()
    window.after(1000, start_bot)

# ========== 📋 讀取輸出 ==========
def read_output():
    while bot_process and bot_process.stdout:
        for line in iter(bot_process.stdout.readline, ''):
            log_output.insert(tk.END, line)
            log_output.see(tk.END)

# ========== 🖼️ UI 設計 ==========
window = tk.Tk()
window.title("戀芯 AI 控制台")
window.geometry("750x500")
window.configure(bg="#ffe4ec")

frame = tk.Frame(window, bg="#ffe4ec")
frame.pack(pady=10)

btn_style = {
    "bg": "#ff8da0",
    "fg": "white",
    "font": ("微軟正黑體", 10, "bold"),
    "relief": tk.RAISED,
    "bd": 2,
    "width": 15,
    "height": 2
}

tk.Button(frame, text="▶️ 啟動機器人", command=start_bot, **btn_style).pack(side=tk.LEFT, padx=10)
tk.Button(frame, text="⛔ 關閉機器人", command=stop_bot, **btn_style).pack(side=tk.LEFT, padx=10)
tk.Button(frame, text="🔄 重新啟動", command=restart_bot, **btn_style).pack(side=tk.LEFT, padx=10)

log_output = ScrolledText(window, height=25, font=("Consolas", 10), bg="white")
log_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# 關閉程式時結束子程序
def on_close():
    global flask_process, bot_process
    if flask_process:
        flask_process.terminate()
    if bot_process:
        bot_process.terminate()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_close)

# 啟動記憶 UI（用 thread 防止卡 UI）
threading.Thread(target=launch_memory_ui, daemon=True).start()

# 主視窗進入事件循環
window.mainloop()
