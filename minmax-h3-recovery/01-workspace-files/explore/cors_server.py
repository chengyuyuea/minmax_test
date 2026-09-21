import http.server, socketserver, os

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

os.chdir("/mnt/workspace/explore")
with socketserver.TCPServer(("", 8765), CORSHandler) as httpd:
    print("CORS server on :8765", flush=True)
    httpd.serve_forever()
