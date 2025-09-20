from fastapi import FastAPI, HTTPException, Query
from database import get_db_connection, init_db
from models import TaskCreate, TaskUpdate
from typing import Optional, List

app = FastAPI()

# 确保数据库已初始化
init_db()

# Create - 添加任务


@app.post("/tasks")
def create_task(task: TaskCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tasks (title, description, completed, priority, due_date) VALUES (?, ?, ?, ?, ?)",
        (task.title, task.description, False, task.priority, task.due_date),
    )
    conn.commit()
    task_id = cursor.lastrowid
    conn.close()
    return {"id": task_id,
            "title": task.title,
            "description": task.description,
            "completed": False,
            "priority": task.priority,
            "due_date": task.due_date, }

# 🔵 Read - 获取所有任务


# @app.get("/tasks")
# def read_tasks(sort_by: str = Query("id", regex="^(id|priority|due_date)$"),
#                order: str = Query("asc", regex="^(asc|desc)$")):
#     conn = get_db_connection()
#     sql = f"SELECT * FROM tasks ORDER BY {sort_by} {order.upper()}"
#     tasks = conn.execute(sql).fetchall()
#     conn.close()
#     return [dict(task) for task in tasks]


# 🔵 Read - 获取所有任务 (已集成过滤、排序、搜索和分页)
@app.get("/tasks", response_model=List[dict])  # 建议添加 response_model
def read_tasks(
    # 过滤参数
    max_due_date: Optional[str] = None,
    # 搜索参数
    keyword: Optional[str] = None,
    # 排序参数
    sort_by: str = Query("id", regex="^(id|priority|due_date)$"),
    order: str = Query("asc", regex="^(asc|desc)$"),
    # 分页参数
    # 使用 Query 添加校验：limit 最小为1，最大为100；offset 最小为0
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    conn = get_db_connection()

    # --- 动态 SQL 构建 ---

    # 基础查询
    query = "SELECT * FROM tasks"
    params = []

    # 1. 构建 WHERE 子句
    where_clauses = []
    if max_due_date:
        where_clauses.append("due_date <= ?")
        params.append(max_due_date)

    if keyword:
        # 在 title 和 description 中进行模糊搜索
        where_clauses.append("(title LIKE ? OR description LIKE ?)")
        # 添加两次参数，一次给 title，一次给 description
        params.extend([f"%{keyword}%", f"%{keyword}%"])

    # 如果有任何 WHERE 条件，将它们用 AND 连接并添加到主查询中
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    # 2. 添加 ORDER BY 子句
    query += f" ORDER BY {sort_by} {order.upper()}"

    # 3. 添加 LIMIT 和 OFFSET 子句 (分页)
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    # --- 执行查询 ---

    cursor = conn.cursor()
    print(f"Executing SQL: {query}")  # 调试时打印最终的 SQL 很有用
    print(f"With Params: {params}")  # 调试时打印参数

    cursor.execute(query, tuple(params))  # 确保 params 是元组
    tasks = cursor.fetchall()

    conn.close()
    return [dict(task) for task in tasks]

# # 🔵 Read - 获取所有任务 (已合并过滤和排序功能)
# @app.get("/tasks")
# def read_tasks(
#     # 将过滤条件作为可选的查询参数
#     max_due_date: Optional[str] = None,
#     # 保留原有的排序参数
#     sort_by: str = Query("id", regex="^(id|priority|due_date)$"),
#     order: str = Query("asc", regex="^(asc|desc)$")
# ):
#     conn = get_db_connection()

#     # 1. 基础查询语句
#     query = "SELECT * FROM tasks"
#     params = []  # 用于安全地传递参数，防止 SQL 注入

#     # 2. 如果提供了过滤条件，则动态添加 WHERE 子句
#     if max_due_date:
#         query += " WHERE due_date <= ?"
#         params.append(max_due_date)

#     # 3. 动态添加排序子句
#     # 注意：直接用 f-string 格式化列名和排序方向是安全的，
#     # 因为我们已经通过 Query 的 regex 严格限制了这两个参数的可能值。
#     query += f" ORDER BY {sort_by} {order.upper()}"

#     # 4. 执行最终构建好的查询
#     cursor = conn.cursor()
#     cursor.execute(query, params)
#     tasks = cursor.fetchall()

#     conn.close()
#     return [dict(task) for task in tasks]


# 🟡 Read - 获取单个任务


@app.get("/tasks/{task_id}")
def read_task(task_id: int):
    conn = get_db_connection()
    task = conn.execute("SELECT * FROM tasks WHERE id = ?",
                        (task_id,)).fetchone()
    conn.close()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return dict(task)

# 🟠 Update - 修改任务


@app.put("/tasks/{task_id}")
def update_task(task_id: int, task_update: TaskUpdate):
    conn = get_db_connection()
    cursor = conn.cursor()

    existing = cursor.execute(
        "SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if existing is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")

    # 如果某个字段没有传，保留原来的值
    title = task_update.title if task_update.title is not None else existing["title"]
    description = task_update.description if task_update.description is not None else existing[
        "description"]
    completed = task_update.completed if task_update.completed is not None else existing[
        "completed"]

    cursor.execute(
        "UPDATE tasks SET title = ?, description = ?, completed = ? WHERE id = ?",
        (title, description, completed, task_id),
    )
    conn.commit()
    conn.close()

    return {"id": task_id, "title": title, "description": description, "completed": completed}

# 🔴 Delete - 删除任务


@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    deleted = cursor.rowcount
    conn.close()

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": f"Task {task_id} deleted"}
