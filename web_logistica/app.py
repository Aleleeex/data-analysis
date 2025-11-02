import http.server
import socketserver
import urllib.parse
import joblib
import os

PORT = 8081  # Usa 8081 para evitar conflicto si ya ejecutas web_salario en 8080
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'Laboratorios 9C', 'modelo_regresion_logistica.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'Laboratorios 9C', 'scaler.pkl')
HTML_PATH = os.path.join(BASE_DIR, 'index.html')

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
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = urllib.parse.parse_qs(post_data.decode('utf-8'))
            try:
                edad = float(data.get('edad', [0])[0])
                salario = float(data.get('salario', [0])[0])

                # Cargar scaler y modelo
                scaler = joblib.load(MODEL_PATH.replace('modelo_regresion_logistica.pkl', 'scaler.pkl')) if os.path.exists(SCALER_PATH) else None
                if scaler is None:
                    scaler = joblib.load(SCALER_PATH)  # fallback normal
                modelo = joblib.load(MODEL_PATH)

                # Escalar y predecir
                X = [[edad, salario]]
                X_scaled = scaler.transform(X) if scaler is not None else X
                pred = int(modelo.predict(X_scaled)[0])

                # Probabilidad (si el modelo la soporta)
                prob_txt = ''
                try:
                    proba = float(modelo.predict_proba(X_scaled)[0][1])
                    prob_txt = f" con probabilidad {proba*100:.1f}%"
                except Exception:
                    pass

                if pred == 1:
                    resultado = f"<h3>Resultado: <span style='color:#10a37f;'>Compraría</span>{prob_txt}</h3>"
                else:
                    resultado = f"<h3>Resultado: <span style='color:#e5534b;'>No compraría</span>{prob_txt}</h3>"

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
    with socketserver.TCPServer(('', PORT), SimpleHTTPRequestHandler) as httpd:
        print(f"Servidor corriendo en http://localhost:{PORT}")
        httpd.serve_forever()
