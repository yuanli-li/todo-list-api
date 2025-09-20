import requests

BASE_URL = "http://127.0.0.1:8000"

def create_task(title, description=None, priority=1, due_date=None):
    payload = {
        "title": title,
        "description": description,
        "priority": priority,
        "due_date": due_date
    }
    r = requests.post(f"{BASE_URL}/tasks", json=payload)
    print("CREATE:", r.status_code, r.json())
    return r.json()["id"]

def get_tasks(sort_by="id", order="asc"):
    r = requests.get(f"{BASE_URL}/tasks", params={"sort_by": sort_by, "order": order})
    print(f"GET ALL sorted by {sort_by} {order.upper()}:", r.status_code)
    for task in r.json():
        print(task)

def filter_tasks(max_due_date):
    r = requests.get(f"{BASE_URL}/tasks/filter", params={"max_due_date": max_due_date})
    print(f"FILTER tasks due before {max_due_date}:", r.status_code)
    for task in r.json():
        print(task)

if __name__ == "__main__":
    # 创建几个任务
    id1 = create_task("Task A", "Desc A", priority=2, due_date="2025-09-25")
    id2 = create_task("Task B", "Desc B", priority=1, due_date="2025-09-20")
    id3 = create_task("Task C", "Desc C", priority=3, due_date="2025-10-01")

    # 获取所有任务，按优先级升序
    get_tasks(sort_by="priority", order="asc")

    # 获取所有任务，按截止日期降序
    get_tasks(sort_by="due_date", order="desc")

    # 筛选截止日期之前的任务
    filter_tasks(max_due_date="2025-09-30")
