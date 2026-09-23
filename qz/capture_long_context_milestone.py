"""Preserve the200-update diagnostic checkpoint without changing the worker.

The pinned worker writes a new temporary file and atomically replaces resume.pt.
A hard link therefore preserves the200-update inode when600 later replaces it;
it does not add a large checkpoint copy to the running training job's I/O.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-plan',type=Path,required=True)
    ap.add_argument('--hours',type=float,default=6);ap.add_argument('--state-name',default='milestone_capture.json');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    if Path(args.state_name).name!=args.state_name:raise ValueError('state name must be a basename')
    state_path=root/'results/long_context_mvp'/args.state_name
    state=dict(status='waiting_for_step200',training_plan=str(args.training_plan.resolve()),started_unix=time.time())
    def save():
        state['updated_unix']=time.time();tmp=state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    deadline=time.monotonic()+args.hours*3600
    try:
        while time.monotonic()<deadline:
            if args.training_plan.exists():
                plan=json.loads(args.training_plan.read_text());source=Path(plan['out'])/'resume.pt'
                if source.exists():
                    destination=Path(plan['out']+'_step200_archive')/'resume.pt'
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if not destination.exists():
                        os.link(source,destination)
                    import torch
                    payload=torch.load(destination,map_location='cpu',weights_only=False,mmap=True)
                    step=payload['step'];contract=payload['contract_sha256'];del payload
                    if step!=200:
                        destination.unlink()
                        state.update(status='milestone_not_captured',observed_checkpoint_step=step)
                        save();return
                    state.update(status='captured',step=200,checkpoint=str(destination),contract_sha256=contract,
                                 source=str(source),source_inode_at_capture=destination.stat().st_ino,
                                 method='Hard link; pinned worker replaces checkpoints atomically, never modifies this inode in place',
                                 scope='Development diagnostic only; terminal600 remains the primary checkpoint')
                    save();return
            save();time.sleep(15)
        state['status']='capture_timeout';save()
    except Exception as error:
        state.update(status='needs_attention',error=str(error));save();raise


if __name__=='__main__':main()
