import argparse
import pandas as pd
from .engine import run
from .report import write_report

def main():
    p=argparse.ArgumentParser(description='Search bounded failure scenarios and verify a repair candidate.')
    p.add_argument('--csv',help='Optional numeric CSV; target must be 0/1')
    p.add_argument('--target',default='target');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trials',type=int,default=60);p.add_argument('--backend',choices=['random','optuna'],default='random')
    p.add_argument('--output',default='reports/latest');args=p.parse_args()
    try:
        x=y=None;source='synthetic demonstration (not real customer data)'
        if args.csv:
            df=pd.read_csv(args.csv)
            if args.target not in df: raise ValueError('Target column not found')
            y=df.pop(args.target);x=df;source='user-supplied numeric CSV'
        result=run(x,y,args.seed,args.trials,args.backend,source)
        path=write_report(result,args.output)
        print(f"Clean balanced accuracy: {result['baseline']['balanced_accuracy']:.3f}")
        print(f"Stress balanced accuracy: {result['stressed']['balanced_accuracy']:.3f}")
        print(f"Repair candidate: {result['repair_stressed']['balanced_accuracy']:.3f}")
        print(f"Repair selected on validation: {result['repair_selected_on_validation']}")
        print(f'Open {path.resolve()} in your browser.')
    except (ValueError,FileNotFoundError,ImportError) as e:
        p.exit(2,f'PreMortemML: {e}\n')
if __name__=='__main__':main()
