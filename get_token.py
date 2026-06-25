"""One-time OAuth flow to get a cTrader Open API access/refresh token.

Run once: uv run python3 get_token.py
Opens your browser to log in and approve access, catches the redirect,
exchanges the code for tokens, and writes them straight into .env.
"""

import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv
import os

from ctrader_open_api import Auth

load_dotenv()

CLIENT_ID = os.environ["CTRADER_CLIENT_ID"]
CLIENT_SECRET = os.environ["CTRADER_CLIENT_SECRET"]
REDIRECT_URI = os.environ.get("CTRADER_REDIRECT_URI", "http://localhost:8080")

auth = Auth(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)


def save_tokens(access_token: str, refresh_token: str) -> None:
    with open(".env") as f:
        lines = f.readlines()

    lines = [
        line
        for line in lines
        if not line.startswith("CTRADER_ACCESS_TOKEN=")
        and not line.startswith("CTRADER_REFRESH_TOKEN=")
    ]
    lines.append(f"CTRADER_ACCESS_TOKEN={access_token}\n")
    lines.append(f"CTRADER_REFRESH_TOKEN={refresh_token}\n")

    with open(".env", "w") as f:
        f.writelines(lines)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if "code=" not in self.path:
            # Stray request (e.g. favicon.ico) -- ignore, keep waiting for the real redirect.
            self.send_response(404)
            self.end_headers()
            return

        code = self.path.split("code=")[1].split("&")[0]
        token_data = auth.getToken(code)

        if "accessToken" not in token_data:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"Token exchange failed: {token_data}".encode())
            print(f"Token exchange failed: {token_data}")
            return

        save_tokens(token_data["accessToken"], token_data["refreshToken"])
        self.server.got_code = True

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Done -- tokens saved to .env. You can close this tab.")
        print("Access and refresh tokens saved to .env")

    def log_message(self, format, *args):
        pass


def main() -> None:
    auth_uri = auth.getAuthUri()
    print(f"Opening browser to:\n{auth_uri}\n")
    print("Log in and approve access. Waiting for redirect on", REDIRECT_URI)
    webbrowser.open(auth_uri)

    server = HTTPServer(("localhost", 8080), Handler)
    server.got_code = False
    while not server.got_code:
        server.handle_request()


if __name__ == "__main__":
    main()
