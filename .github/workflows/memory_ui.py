# memory_ui.py
from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os

app = Flask(__name__, template_folder="templates")
DB_PATH = 'lianxin_ai.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    query = request.args.get('q', '')
    conn = get_db()
    if query:
        memories = conn.execute("SELECT * FROM memories WHERE content LIKE ?", ('%' + query + '%',)).fetchall()
    else:
        memories = conn.execute("SELECT * FROM memories ORDER BY id DESC").fetchall()
    return render_template('index.html', memories=memories, query=query)

@app.route('/add', methods=['POST'])
def add_memory():
    content = request.form['content']
    user_id = request.args.get("user_id", os.getenv("USER_ID", "default_user"))
    conn = get_db()
    conn.execute("INSERT INTO memories (user_id, role, content) VALUES (?, ?, ?)", (user_id, 'memory', content))
    conn.commit()
    return redirect(url_for('index', user_id=user_id))

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_memory(id):
    user_id = request.args.get("user_id", os.getenv("USER_ID", "default_user"))
    conn = get_db()
    if request.method == 'POST':
        content = request.form['content']
        conn.execute("UPDATE memories SET content = ? WHERE id = ?", (content, id))
        conn.commit()
        return redirect(url_for('index', user_id=user_id))
    memory = conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
    return render_template('edit.html', memory=memory)

@app.route('/delete/<int:id>')
def delete_memory(id):
    user_id = request.args.get("user_id", os.getenv("USER_ID", "default_user"))
    conn = get_db()
    conn.execute("DELETE FROM memories WHERE id = ?", (id,))
    conn.commit()
    return redirect(url_for('index', user_id=user_id))

@app.route('/user_profile', methods=['GET', 'POST'])
def user_profile():
    user_id = request.args.get("user_id", os.getenv("USER_ID", "default_user"))
    conn = get_db()
    if request.method == 'POST':
        nickname = request.form.get('nickname')
        age = request.form.get('age')
        gender = request.form.get('gender')
        background = request.form.get('background')
        extra = request.form.get('extra')

        existing = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
        if existing:
            conn.execute("""
                UPDATE user_profiles SET nickname=?, age=?, gender=?, background=?, extra=? WHERE user_id=?
            """, (nickname, age, gender, background, extra, user_id))
        else:
            conn.execute("""
                INSERT INTO user_profiles (user_id, nickname, age, gender, background, extra)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, nickname, age, gender, background, extra))
        conn.commit()
        return redirect(url_for('user_profile', user_id=user_id))

    profile = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    return render_template('user_profile.html', profile=profile)

@app.route('/character_profile', methods=['GET', 'POST'])
def character_profile():
    user_id = request.args.get("user_id", os.getenv("USER_ID", "default_user"))
    conn = get_db()
    if request.method == 'POST':
        form = request.form
        fields = ["name", "age", "occupation", "relationship", "background",
                  "personality", "speaking_style", "likes", "dislikes", "extra"]
        values = [form.get(f) for f in fields]

        existing = conn.execute("SELECT * FROM characters WHERE user_id = ?", (user_id,)).fetchone()
        if existing:
            set_clause = ", ".join(f"{f}=?" for f in fields)
            conn.execute(f"UPDATE characters SET {set_clause} WHERE user_id=?", (*values, user_id))
        else:
            conn.execute("""
                INSERT INTO characters (user_id, name, age, occupation, relationship, background,
                    personality, speaking_style, likes, dislikes, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, *values))
        conn.commit()
        return redirect(url_for('character_profile', user_id=user_id))

    character = conn.execute("SELECT * FROM characters WHERE user_id = ?", (user_id,)).fetchone()
    return render_template('character_profile.html', character=character)

if __name__ == '__main__':
    app.run(debug=True)
