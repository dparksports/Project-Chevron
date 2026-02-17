"""
TodoStore — In-memory store for todo items. Pure data layer.
SCP Glyph: ◬, Ө, ☾
Dependencies: none
"""

class TodoStore:
    def add_todo(self, text: str) -> Todo:
        "Origin — creates a new todo item"
        raise NotImplementedError

    def get_todo(self, todo_id: int) -> Todo | None:
        "Filter — returns todo by ID or None"
        raise NotImplementedError

    def list_todos(self, filter: str) -> list[Todo]:
        "Filter — returns todos matching filter (all/active/done)"
        raise NotImplementedError

    def toggle_todo(self, todo_id: int) -> Todo:
        "Fold — toggles completion state"
        raise NotImplementedError

    def delete_todo(self, todo_id: int) -> bool:
        "Fold — removes a todo item"
        raise NotImplementedError

