import tkinter as tk
from tkinter import messagebox

def save_env():
    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"DISCORD_TOKEN={token_entry.get()}\n")
        f.write(f"OPENROUTER_API_KEY={api_entry.get()}\n")
        f.write(f"USER_ID={user_id_entry.get()}\n")
        f.write(f"ORDER_CODE={order_code_entry.get()}\n")
    messagebox.showinfo("完成", "✅ .env 檔案已建立，可啟動機器人")
    window.destroy()

window = tk.Tk()
window.title("戀芯 AI 安裝精靈")

labels = ["DISCORD_TOKEN", "OPENROUTER_API_KEY", "USER_ID", "ORDER_CODE"]
entries = []

for label_text in labels:
    tk.Label(window, text=label_text).pack()
    entry = tk.Entry(window, width=50)
    entry.pack()
    entries.append(entry)

token_entry, api_entry, user_id_entry, order_code_entry = entries

tk.Button(window, text="✅ 建立 .env", command=save_env).pack(pady=10)

window.mainloop()
