import threading
import webbrowser

from app import app

HOST = "127.0.0.1"
PORT = 5000


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)