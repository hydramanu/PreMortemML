import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from premortem import server
from premortem.engine import demo_data

@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(server,'DATA',tmp_path)
    server.initialize()
    http = server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
    thread = threading.Thread(target=http.serve_forever,daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{http.server_port}'
    http.shutdown()
    thread.join()
    http.server_close()

def post(api,body,**headers):
    return urlopen(Request(api+'/api/runs',data=json.dumps(body).encode(),headers={'Content-Type':'application/json',**headers}))

def test_demo_and_downloads(api):
    rows=json.load(urlopen(api+'/api/runs'))
    assert rows[0]['id']=='demo'
    result=json.load(urlopen(api+'/api/runs/demo/result'))
    assert result['trials']==80
    for kind in ['report','json','csv']:
        response=urlopen(api+'/api/runs/demo/'+kind)
        assert response.status==200 and len(response.read())>100
        assert 'attachment' in response.headers['Content-Disposition']

@pytest.mark.parametrize('payload',[{'trials':0},{'seed':-1},{'backend':'invalid'},{'source':'csv'},{'csv':'a,b\n1,0','target':'absent'}])
def test_invalid_inputs(api,payload):
    with pytest.raises(HTTPError) as error: post(api,payload)
    assert error.value.code==400

def test_cross_origin_rejected(api):
    with pytest.raises(HTTPError) as error: post(api,{},Origin='http://unrelated.example')
    assert error.value.code==403

def test_csv_and_saved_run(api,monkeypatch):
    # Execute synchronously for deterministic API integration coverage.
    class Immediate:
        def submit(self,fn,*args): fn(*args)
    monkeypatch.setattr(server,'POOL',Immediate())
    x,y=demo_data(n=300)
    x['outcome']=y
    response=post(api,{'name':'CSV integration','trials':5,'source':'csv','csv':x.to_csv(index=False),'target':'outcome','filename':'test.csv'})
    identifier=json.load(response)['id']
    result=json.load(urlopen(api+f'/api/runs/{identifier}/result'))
    assert result['source']=='test.csv'
    assert sum(result['split_rows'].values())==300
    server.initialize()
    rows=json.load(urlopen(api+'/api/runs'))
    assert any(r['id']==identifier and r['status']=='completed' for r in rows)

def test_interrupted_run_recovery(api):
    with server.connection() as db:
        db.execute("INSERT INTO runs(id,name,status,config) VALUES('interrupted','Interrupted','running','{}')")
    server.initialize()
    with server.connection() as db:
        assert db.execute("SELECT status FROM runs WHERE id='interrupted'").fetchone()[0]=='failed'

def test_queue_capacity(api):
    with server.connection() as db:
        for i in range(3):
            db.execute('INSERT INTO runs(id,name,status,config) VALUES(?,?,?,?)',(str(i),'Queued','queued','{}'))
    with pytest.raises(HTTPError) as error: post(api,{'trials':5})
    assert error.value.code==429

def test_worker_failure_is_persisted(api,monkeypatch):
    with server.connection() as db:
        db.execute("INSERT INTO runs(id,name,status,config) VALUES('failure','Failure','queued','{}')")
    def fail(*args,**kwargs): raise ValueError('Deliberate test failure')
    monkeypatch.setattr(server,'run',fail)
    server.execute('failure',{'seed':42,'trials':5,'backend':'random','source':'test'},None,None)
    with server.connection() as db:
        row=db.execute("SELECT status,error FROM runs WHERE id='failure'").fetchone()
    assert row==('failed','Deliberate test failure')
