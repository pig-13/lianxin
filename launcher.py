# ── launcher.py  (Thread 直跑 bot.py / memory_ui.py) ──
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading, runpy, os, sys
from env_setup import create_env          # ⬅ 如有 .env 自動產生

# ───────── 基本工具 ─────────
def base_path():                       # 對應 PyInstaller 的 _MEIPASS
    return getattr(sys, "_MEIPASS", os.path.abspath("."))

def rpath(rel):                        # 取相對路徑
    return os.path.join(base_path(), rel)

def log(msg):
    log_area.insert(tk.END, msg + "\n")
    log_area.see(tk.END)

# ───────── 執行腳本：用 Thread + runpy ─────────
def run_script(rel_path, tag):
    def _target():
        script = rpath(rel_path)
        runpy.run_path(script, run_name="__main__")
    t = threading.Thread(target=_target, name=tag, daemon=True)
    t.start()
    return t

bot_thread = ui_thread = None

# ───────── 按鈕回呼 ─────────
def start_services():
    global bot_thread, ui_thread
    if bot_thread or ui_thread:
        log("⚠️ 服務已在運行")
        return
    try:
        bot_thread = run_script("bot.py", "BOT")
        ui_thread  = run_script("memory_ui.py", "MEM")
        log("✅ 機器人 & 記憶系統啟動成功")
    except Exception as e:
        log(f"❌ 啟動失敗：{e}")

def stop_services():
    log("⚠️ Thread 方式無法強制關閉腳本，請直接關閉視窗結束。")

# ───────── UI ─────────
create_env()                # 生成 .env（首次執行才會寫入）

win = tk.Tk()
win.title("戀芯控制台")
win.geometry("500x400")
win.configure(bg="#ffe6f0")

tk.Label(win, text="戀芯 AI 啟動器",
        bg="#ffe6f0", fg="#d6336c", font=("微軟正黑體", 20)).pack(pady=10)
tk.Button(win, text="啟動機器人", command=start_services,
        bg="#ffb6c1", font=("微軟正黑體", 14)).pack(pady=10)
tk.Button(win, text="關閉機器人", command=stop_services,
        bg="#ffc0cb", font=("微軟正黑體", 14)).pack(pady=5)

log_area = ScrolledText(win, height=10, font=("Courier New", 10))
log_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

win.protocol("WM_DELETE_WINDOW", win.destroy)
win.mainloop()
