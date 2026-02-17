"""
TodoAPI — REST API layer. Routes HTTP requests to TodoStore.
SCP Glyph: ☤
Dependencies: TodoStore
"""

class TodoAPI:
    def handle_request(self, method: str, path: str, body: dict) -> Response:
        "Weaves HTTP request into response via TodoStore"
        raise NotImplementedError

