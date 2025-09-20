import sqlite3


def get_db_connection():
    conn = sqlite3.connect("todo.db")
    conn.row_factory = sqlite3.Row  # 返回 dict 风格
    return conn

# 初始化数据库（只需要执行一次）


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        completed BOOLEAN NOT NULL DEFAULT 0,
        priority INTEGER NOT NULL DEFAULT 1,
        due_date TEXT
    )
    ''')
    conn.commit()
    conn.close()


# 如果你直接运行 database.py，可以初始化数据库
if __name__ == "__main__":
    init_db()
