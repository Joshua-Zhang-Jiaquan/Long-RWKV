"""Wait on admission races without retrying any scheduler submission."""
import json
import time
from confirm_capacity import CapacityUnavailable
from submission_lock import campaign_submission_lock


def run_when_capacity(root, action, *, sleep=time.sleep, lock=campaign_submission_lock):
    while True:
        try:
            with lock(root):
                return action()
        except CapacityUnavailable as error:
            # The lock is released before waiting, so other controllers can run.
            # Every other error escapes, including ambiguous submission failures.
            print(json.dumps(dict(status='waiting_for_capacity',reason=str(error))),flush=True)
            sleep(30)
