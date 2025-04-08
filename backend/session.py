from itsdangerous import URLSafeSerializer
from fastapi import Request, Response

class SessionManager:
    def __init__(self, secret_key: str):
        self.serializer = URLSafeSerializer(secret_key)
        self.cookie_name = "session"

    def get_session(self, request: Request) -> dict:
        session_cookie = request.cookies.get(self.cookie_name)
        if not session_cookie:
            print("No session cookie found")
            return {}
        try:
            data = self.serializer.loads(session_cookie)
            print("Session data:", data)
            return data
        except Exception as e:
            print(f"Error loading session: {str(e)}")
            return {}

    def set_session(self, response: Response, data: dict):
        try:
            session_data = self.serializer.dumps(data)
            response.set_cookie(
                key=self.cookie_name,
                value=session_data,
                httponly=True,
                samesite='lax',
                secure=False,  # Development için secure=False
                path="/"  # Tüm path'lerde geçerli
            )
            print("Session set successfully")
        except Exception as e:
            print(f"Error setting session: {str(e)}")

    def clear_session(self, response: Response):
        try:
            response.delete_cookie(
                key=self.cookie_name,
                path="/"
            )
            print("Session cleared successfully")
        except Exception as e:
            print(f"Error clearing session: {str(e)}")

# Daha güvenli bir secret key kullanıyoruz
session_manager = SessionManager("moodmap-secret-key-2024") 