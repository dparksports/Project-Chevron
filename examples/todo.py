from typing import List, Dict, TypedDict

# Define data structures
class Task(TypedDict):
    id: str
    description: str
    completed: bool

Store = List[Task]

# ◬ TodoStore Module
class TodoStore:

    @staticmethod
    def add(task: Task, store: Store) -> Store:
        """☤ Weaves task into store"""
        # ☤ Weaver
        return store + [task]

    @staticmethod
    def remove(task_id: str, store: Store) -> Store:
        """Ө Filters out the task matching the ID"""
        # Ө The Filter
        return [task for task in store if task['id'] != task_id]

    @staticmethod
    def list(store: Store) -> List[Task]:
        """𓂀 Witnesses all tasks without modification"""
        # 𓂀 The Witness
        return store

    @staticmethod
    def complete(task_id: str, store: Store) -> Store:
        """☾ Folds task state from incomplete → complete"""
        # ☾ Fold Time
        def complete_task(task: Task) -> Task:
            if task['id'] == task_id:
                return Task(id=task['id'], description=task['description'], completed=True)
            else:
                return task

        return [complete_task(task) for task in store]