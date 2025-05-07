from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os


def get_google_drive_credentials(credentials_path: str, token_path: str = 'token.json') -> Credentials:
    """Генерирует или загружает учетные данные для Google Drive API."""
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None

    # Проверяем, существует ли token.json
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    # Если нет валидных учетных данных, обновляем или создаем новые
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)
        # Сохраняем учетные данные в token.json
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    return creds


if __name__ == "__main__":
    # Пример использования
    credentials_path = '../credentials.json'
    creds = get_google_drive_credentials(credentials_path)
    print("Credentials generated successfully.")
# Комментарий: Этот модуль отвечает за аутентификацию в Google Drive API. Функция get_google_drive_credentials загружает или генерирует учетные данные, используя credentials.json и token.json. Поддерживает обновление токенов и запуск локального сервера для OAuth. Используется в gdrive_extractor.py.