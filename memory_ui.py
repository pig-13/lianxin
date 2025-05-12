# memory_ui.py
from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)
DB_PATH = 'lianxin_ai.db'  # 修改為你實際的資料庫路徑

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
    conn = get_db()
    conn.execute("INSERT INTO memories (user_id, role, content) VALUES (?, ?, ?)", ('your_user_id', 'memory', content))
    conn.commit()
    return redirect(url_for('index'))

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_memory(id):
    conn = get_db()
    if request.method == 'POST':
        content = request.form['content']
        conn.execute("UPDATE memories SET content = ? WHERE id = ?", (content, id))
        conn.commit()
        return redirect(url_for('index'))
    memory = conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
    return render_template('edit.html', memory=memory)

@app.route('/delete/<int:id>')
def delete_memory(id):
    conn = get_db()
    conn.execute("DELETE FROM memories WHERE id = ?", (id,))
    conn.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
