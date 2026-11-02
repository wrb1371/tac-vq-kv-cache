"""CPU verification of every actual calibration read in all three replicas."""
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ar39cal.bindings import audit
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--torch',action='store_true');a=ap.parse_args()
    rows=[audit(r,a.torch)for r in range(3)]
    print(json.dumps({'actual_file_bindings_verified':True,'replicas':len(rows),'files':sum(len(r['files'])for r in rows)}))
