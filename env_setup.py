import os
import tkinter as tk
from tkinter import simpledialog, messagebox

def create_env():
    if os.path.exists(".env"):
        return  # 已存在就跳過

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("初始設定（戀芯）", "歡迎使用戀芯桌面版！\n\n第一次使用前，請輸入以下資訊：")

    fields = {
        "OPENROUTER_API_KEY": "請輸入 OpenRouter API Key",
        "DISCORD_TOKEN":      "請輸入 Discord Bot Token",
        "ORDER_CODE":         "請輸入購買訂單編碼",
        "USER_DC_ID":         "請輸入您的 Discord 使用者 ID"
    }

    lines = []
    for k, prompt in fields.items():
        v = simpledialog.askstring("戀芯設定", prompt, parent=root)
        if not v:
            messagebox.showerror("取消", "資料未填完整，程式結束")
            root.destroy()
            exit(1)
        lines.append(f"{k}={v.strip()}\n")

    with open(".env", "w", encoding="utf-8") as f:
        f.writelines(lines)

    messagebox.showinfo("完成", ".env 已建立完成！")
    root.destroy()
