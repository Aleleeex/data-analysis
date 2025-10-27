import http.server
import socketserver
import urllib.parse
import joblib
import os

PORT = 8080
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Laboratorios 9C', 'modelo_salario.pkl')
HTML_PATH = os.path.join(os.path.dirname(__file__), 'index.html')

class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/index.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            with open(HTML_PATH, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, 'Archivo no encontrado')

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = urllib.parse.parse_qs(post_data.decode('utf-8'))
            try:
                anios = float(data.get('anios', [0])[0])
                modelo = joblib.load(MODEL_PATH)
                prediccion = modelo.predict([[anios]])[0]
                resultado = f"<h3>Para {anios} años de experiencia, el salario predicho es: <span style='color:green;'>${prediccion:,.2f}</span></h3>"
            except Exception as e:
                resultado = f"<h3 style='color:red;'>Error: {e}</h3>"
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            with open(HTML_PATH, 'r', encoding='utf-8') as f:
                html = f.read()
            html = html.replace('<!-- Aquí aparecerá el resultado -->', resultado)
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_error(404, 'Ruta no encontrada')

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
        print(f"Servidor corriendo en http://localhost:{PORT}")
        httpd.serve_forever()
