import csv
import os

CUTOFF = '2026-07-29 07:00:00'
SOURCE = 'trades_audited.csv'
CLEAN = 'trades_audited_clean.csv'

# читаем свежий полный датасет (после повторного запуска journal_audit.py)
with open(SOURCE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    all_rows = list(reader)

# читаем уже сохранённые id, чтобы не дублировать
existing_ids = set()
if os.path.exists(CLEAN):
    with open(CLEAN, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_ids.add(row['id'])
else:
    existing_ids = set()

new_rows = [
    r for r in all_rows
    if r.get('closed_at', '') > CUTOFF and r['id'] not in existing_ids
]

if new_rows:
    file_exists = os.path.exists(CLEAN)
    with open(CLEAN, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)
    print(f'Добавлено новых сделок: {len(new_rows)}')
else:
    print('Новых сделок нет')

# итоговая статистика
total = len(existing_ids) + len(new_rows)
print(f'Всего в чистой базе: {total}')
