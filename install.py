def ask_and_write_env():
    print("🔧 歡迎使用戀芯 AI 安裝精靈，請輸入以下資訊：\n")

    discord_token = input("1️⃣ 請輸入你的 DISCORD_TOKEN：")
    api_key = input("2️⃣ 請輸入你的 OPENROUTER_API_KEY：")
    user_id = input("3️⃣ 請輸入你的 Discord 使用者 ID：")
    order_code = input("4️⃣ 請輸入你的訂單序號：")

    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"DISCORD_TOKEN={discord_token}\n")
        f.write(f"OPENROUTER_API_KEY={api_key}\n")
        f.write(f"USER_ID={user_id}\n")
        f.write(f"ORDER_CODE={order_code}\n")

    print("\n✅ 設定完成！已產生 .env 檔案，你現在可以執行主程式 `bot.py` 了。")

if __name__ == "__main__":
    ask_and_write_env()
