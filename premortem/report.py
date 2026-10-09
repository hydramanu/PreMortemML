"""Self-contained, offline interactive HTML report. No external assets."""
import json
from pathlib import Path

def write_report(result,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    payload=json.dumps(result,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8')
    (output/'dashboard.html').write_text(template.replace('__RESULT_JSON__',payload),encoding='utf-8')
    import pandas as pd
    pd.DataFrame(result['history']).to_csv(output/'search_history.csv',index=False)
    return output/'dashboard.html'
