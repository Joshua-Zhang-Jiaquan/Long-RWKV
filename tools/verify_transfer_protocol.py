"""CPU-only verification of the prospective protocol and scientific source freeze."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    path=ROOT/'revision/transfer/FROZEN.json';protocol=json.loads(path.read_text())
    assert protocol['status']=='frozen'
    for field in ('scientific_sources','analysis_sources'):
        for relative,digest in protocol[field].items():
            if sha(ROOT/relative)!=digest:raise ValueError('frozen source changed: '+relative)
    calibration=protocol['calibration_panel'];heldout=protocol['heldout_panel']
    assert len(calibration)==180 and len(heldout)==240
    assert {c['class_id'] for c in calibration}==set(range(15))
    assert {c['class_id'] for c in heldout}==set(range(15,25))
    assert len(protocol['training_seeds'])==len(set(protocol['training_seeds']))==6
    assert protocol['training']['steps']==3100
    assert protocol['interpretation']['both_policies_D']==0
    assert protocol['interpretation']['calls_each']==2
    print(json.dumps(dict(status='passed',protocol_sha256=sha(path),source_files=sum(len(protocol[x]) for x in ('scientific_sources','analysis_sources')),
                          calibration_conditions_per_model=len(calibration),heldout_conditions_per_model=len(heldout),lineages=6),indent=2))
if __name__=='__main__':main()
