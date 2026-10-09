"""Local web workspace with persistent experiments and a single background worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sqlite3
import uuid
from urllib.parse import urlparse
import pandas as pd
from .engine import run, validate
from .report import write_report

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'workspace'
POOL = ThreadPoolExecutor(max_workers=1)

def connection():
    return sqlite3.connect(DATA / 'experiments.db', timeout=30)

def initialize():
    DATA.mkdir(exist_ok=True)
    with connection() as db:
        db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, name TEXT, status TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP, config TEXT, error TEXT)')
        db.execute("UPDATE runs SET status='failed', error='The server stopped before this experiment finished. Start a new run.' WHERE status IN ('running','queued')")
        if not db.execute('SELECT 1 FROM runs').fetchone():
            demo = ROOT / 'reports' / 'demo' / 'results.json'
            if demo.exists():
                result = json.loads(demo.read_text(encoding='utf-8'))
                write_report(result, DATA / 'demo')
                db.execute('INSERT INTO runs(id,name,status,config) VALUES(?,?,?,?)', ('demo','Reference experiment','completed',json.dumps({'trials':result['trials'],'seed':result['seed'],'backend':result['backend']})))

def update(identifier, status, error=None):
    with connection() as db:
        db.execute('UPDATE runs SET status=?,error=? WHERE id=?', (status,error,identifier))

def execute(identifier, config, x, y):
    update(identifier, 'running')
    try:
        result = run(x, y, seed=config['seed'], trials=config['trials'], backend=config['backend'], source=config['source'])
        write_report(result, DATA / identifier)
        update(identifier, 'completed')
    except Exception as exc:
        update(identifier, 'failed', str(exc))

def prepare(payload):
    config = {k: payload.get(k, default) for k,default in [('trials',40),('seed',42),('backend','random')]}
    if type(config['trials']) is not int or not 5 <= config['trials'] <= 500:
        raise ValueError('Trials must be an integer between 5 and 500.')
    if type(config['seed']) is not int or not 0 <= config['seed'] <= 2**32-1000:
        raise ValueError('Seed must be an integer between 0 and 4294966296.')
    if config['backend'] not in ('random','optuna'):
        raise ValueError('Choose random or optuna search.')
    name = str(payload.get('name','Untitled experiment')).strip()[:100] or 'Untitled experiment'
    x = y = None
    config['source'] = 'synthetic demonstration'
    if payload.get('csv'):
        frame = pd.read_csv(io.StringIO(payload['csv']))
        target = payload.get('target')
        if target not in frame.columns:
            raise ValueError('The target column is not in this CSV.')
        if len(frame)>50000 or len(frame.columns)>101:
            raise ValueError('Local workspace supports up to 50,000 rows and 100 features.')
        y = frame[target]
        x = frame.drop(columns=[target])
        validate(x,y)
        config['source'] = str(payload.get('filename','Uploaded CSV'))[:200]
    elif payload.get('source') == 'csv':
        raise ValueError('Select a CSV file before starting.')
    return name,config,x,y

class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, kind='application/json', attachment=None):
        if kind == 'application/json': body = json.dumps(body).encode()
        elif isinstance(body,str): body = body.encode()
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        if attachment: self.send_header('Content-Disposition',f'attachment; filename="{attachment}"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/runs':
            with connection() as db:
                db.row_factory = sqlite3.Row
                rows = [dict(r) for r in db.execute('SELECT * FROM runs ORDER BY created DESC, rowid DESC')]
            for row in rows: row['config'] = json.loads(row['config'])
            return self.reply(200,rows)
        parts = path.strip('/').split('/')
        if len(parts)==4 and parts[:2]==['api','runs']:
            identifier,kind = parts[2:]
            with connection() as db:
                exists = db.execute('SELECT status FROM runs WHERE id=?',(identifier,)).fetchone()
            if not exists: return self.reply(404,{'error':'Experiment not found.'})
            files = {'result':('results.json','application/json'), 'json':('results.json','application/octet-stream'), 'csv':('search_history.csv','text/csv'), 'report':('dashboard.html','text/html; charset=utf-8')}
            if kind not in files: return self.reply(404,{'error':'Unknown export.'})
            filename,mime = files[kind]
            file = DATA / identifier / filename
            if exists[0]!='completed' or not file.exists(): return self.reply(409,{'error':'Results are not ready.'})
            if kind=='result': return self.reply(200,json.loads(file.read_text(encoding='utf-8')))
            return self.reply(200,file.read_bytes(),mime,filename)
        static = {'/':'index.html','/app.js':'app.js','/insights.js':'insights.js','/styles.css':'styles.css'}
        if path in static:
            file = Path(__file__).with_name('web') / static[path]
            mime = {'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[file.suffix[1:]]
            return self.reply(200,file.read_bytes(),mime)
        return self.reply(404,{'error':'Not found.'})

    def do_POST(self):
        if self.path!='/api/runs': return self.reply(404,{'error':'Not found.'})
        # Browsers must send same-origin JSON; no permissive CORS or form submissions.
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':
            return self.reply(415,{'error':'JSON required.'})
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host',''):
            return self.reply(403,{'error':'Cross-origin requests are not allowed.'})
        try:
            length = int(self.headers.get('Content-Length',0))
            if not 0<length<=10_000_000: raise ValueError('Request must be under 10 MB.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload,dict): raise ValueError('Expected a JSON object.')
            name,config,x,y = prepare(payload)
            with connection() as db:
                if db.execute("SELECT COUNT(*) FROM runs WHERE status IN ('queued','running')").fetchone()[0]>=3:
                    return self.reply(429,{'error':'Three experiments are already pending. Wait for a run to finish.'})
                identifier = uuid.uuid4().hex
                db.execute('INSERT INTO runs(id,name,status,config) VALUES(?,?,?,?)',(identifier,name,'queued',json.dumps(config)))
            POOL.submit(execute,identifier,config,x,y)
            return self.reply(202,{'id':identifier})
        except (ValueError,TypeError,KeyError,pd.errors.ParserError) as exc:
            return self.reply(400,{'error':str(exc)})

def main():
    parser = argparse.ArgumentParser(description='Run the PreMortemML local web workspace')
    parser.add_argument('--port',type=int,default=8765)
    args = parser.parse_args()
    initialize()
    server = ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'PreMortemML is running at http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close(); POOL.shutdown(wait=True)

if __name__=='__main__': main()
