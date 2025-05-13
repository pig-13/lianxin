import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading
import os

bot_process = None
flask_process = None

def launch_memory_ui():
    global flask_process
    flask_process = subprocess.Popen(
        ["memory_ui.exe"],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

def start_bot():
    global bot_process
    if bot_process is None:
        log_output.insert(tk.END, "✅ 啟動中...\n")
        bot_process = subprocess.Popen(["bot.exe"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_output.insert(tk.END, "✅ 已啟動戀芯 AI 機器人\n請到 http://localhost:5000 查看記憶介面\n\n")
        threading.Thread(target=read_output, daemon=True).start()

def stop_bot():
    global bot_process
    if bot_process:
        bot_process.terminate()
        log_output.insert(tk.END, "🛑 已關閉戀芯 AI 機器人\n")
        bot_process = None

def restart_bot():
    stop_bot()
    window.after(1000, start_bot)

def read_output():
    while bot_process and bot_process.stdout:
        line = bot_process.stdout.readline()
        if not line:
            break
        log_output.insert(tk.END, line)
        log_output.see(tk.END)

# === UI ===
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

def on_close():
    global flask_process, bot_process
    if flask_process:
        flask_process.terminate()
    if bot_process:
        bot_process.terminate()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_close)

# 新增：用 threading 執行記憶頁面
threading.Thread(target=launch_memory_ui, daemon=True).start()

window.mainloop()
