import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import quant_worker  # importeert main() van de quant worker

_worker_lock = threading.Lock()


def _run_worker_once():
    # Voorkomt dat twee cron-verzoeken tegelijk trainen en elkaars state-bestanden overschrijven
    if not _worker_lock.acquire(blocking=False):
        print("⏳ Worker draait al; verzoek genegeerd.")
        return
    try:
        quant_worker.main()
    except Exception as e:
        print(f"❌ quant_worker crashte: {e}")
    finally:
        _worker_lock.release()


class HealthCheckHandler(BaseHTTPRequestHandler):
    def _reply(self, body: bytes):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def do_GET(self):
        if self.path == '/run-worker':
            if _worker_lock.locked():
                self._reply(b'Worker already running')
            else:
                print("🚀 Cron-verzoek op /run-worker. Start quant_worker op de achtergrond...")
                threading.Thread(target=_run_worker_once, daemon=True).start()
                self._reply(b'Worker Started Successfully')
        else:
            self._reply(b'OK')

    do_HEAD = do_GET

    def log_message(self, format, *args):
        pass


def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"🌐 Server gestart op poort {port}...")
    server.serve_forever()


if __name__ == '__main__':
    run_server()
