import os, re
from datetime import datetime

INTERVAL = 60  # 原版 55+random*10
ONCE = False

script_dir = os.path.dirname(os.path.abspath(__file__))
PENDING = os.path.join(script_dir, '../sche_tasks/pending')

def check():
    if not os.path.isdir(PENDING): return None
    now = datetime.now()
    for f in os.listdir(PENDING):
        m = re.match(r'(\d{4}-\d{2}-\d{2})_(\d{4})_', f)
        if m and now >= datetime.strptime(f'{m[1]} {m[2]}', '%Y-%m-%d %H%M'):
            raw = open(os.path.join(PENDING, f), encoding='utf-8').read()
            return f'按scheduled_task_sop执行任务文件 ../sche_tasks/pending/{f}（立刻移到running）\n内容：\n{raw}'
    return None