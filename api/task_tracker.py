# api/task_tracker.py
# Simple Task Tracker CLI (persists tasks between runs)
# Usage: python api/task_tracker.py

import json
from pathlib import Path

TASKS_FILE = Path(__file__).with_name("tasks.json")


def load_tasks():
    if TASKS_FILE.exists():
        try:
            return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_tasks(tasks):
    TASKS_FILE.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")


def list_tasks(tasks):
    if not tasks:
        print("\nNo tasks yet. Add one!\n")
        return
    print("\nYour Tasks:")
    for i, t in enumerate(tasks, start=1):
        status = "✅" if t.get("done") else "🟡"
        print(f"{i}. {status} {t.get('text')}")
    print("")


def add_task(tasks):
    text = input("Enter task text: ").strip()
    if not text:
        print("Empty task skipped.")
        return
    tasks.append({"text": text, "done": False})
    save_tasks(tasks)
    print("Added!\n")


def mark_done(tasks):
    list_tasks(tasks)
    if not tasks:
        return
    try:
        idx = int(input("Mark which task as done? (number): ").strip())
        if 1 <= idx <= len(tasks):
            tasks[idx - 1]["done"] = True
            save_tasks(tasks)
            print("Marked as done!\n")
        else:
            print("Invalid number.\n")
    except ValueError:
        print("Please enter a valid number.\n")


def clear_tasks():
    confirm = input("Are you sure you want to clear ALL tasks? (y/N): ").strip().lower()
    if confirm == "y":
        save_tasks([])
        print("All tasks cleared.\n")
    else:
        print("Aborted.\n")


def main():
    tasks = load_tasks()
    while True:
        print("=== Task Tracker ===")
        print("1) List tasks")
        print("2) Add task")
        print("3) Mark task as done")
        print("4) Clear all tasks")
        print("5) Exit")
        choice = input("Choose (1-5): ").strip()

        if choice == "1":
            list_tasks(tasks)
        elif choice == "2":
            add_task(tasks)
            tasks = load_tasks()
        elif choice == "3":
            mark_done(tasks)
            tasks = load_tasks()
        elif choice == "4":
            clear_tasks()
            tasks = load_tasks()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.\n")


if __name__ == "__main__":
    main()
