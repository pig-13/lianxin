from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)
import os
import sys

def get_resource_path(relative_path):
    """打包與未打包皆可使用的路徑定位函式"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

DB_PATH = get_resource_path("lianxin_ai.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    user_id = request.args.get("user_id", "").strip()
    query = request.args.get('q', '').strip()
    conn = get_db()

    if query:
        memories = conn.execute("SELECT * FROM memories WHERE user_id = ? AND content LIKE ?", (user_id, f"%{query}%")).fetchall()
    else:
        memories = conn.execute("SELECT * FROM memories WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()

    return render_template('index.html', memories=memories, query=query, user_id=user_id)

@app.route('/add', methods=['POST'])
def add_memory():
    user_id = request.args.get("user_id", "").strip()
    content = request.form['content']
    conn = get_db()
    conn.execute("INSERT INTO memories (user_id, role, content) VALUES (?, ?, ?)", (user_id, 'memory', content))
    conn.commit()
    return redirect(url_for('index', user_id=user_id))

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_memory(id):
    user_id = request.args.get("user_id", "").strip()
    conn = get_db()
    if request.method == 'POST':
        content = request.form['content']
        conn.execute("UPDATE memories SET content = ? WHERE id = ?", (content, id))
        conn.commit()
        return redirect(url_for('index', user_id=user_id))
    memory = conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
    return render_template('edit.html', memory=memory, user_id=user_id)

@app.route('/delete/<int:id>')
def delete_memory(id):
    user_id = request.args.get("user_id", "").strip()
    conn = get_db()
    conn.execute("DELETE FROM memories WHERE id = ?", (id,))
    conn.commit()
    return redirect(url_for('index', user_id=user_id))

@app.route('/user_profile', methods=['GET', 'POST'])
def user_profile():
    user_id = request.args.get("user_id", "").strip()
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
    return render_template('user_profile.html', profile=profile, user_id=user_id)

@app.route('/character_profile', methods=['GET', 'POST'])
def character_profile():
    user_id = request.args.get("user_id", "").strip()
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
    return render_template('character_profile.html', character=character, user_id=user_id)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
