from contextlib import contextmanager
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qz'))
from confirm_capacity import CapacityUnavailable
from distance_admission import run_when_capacity


def test_capacity_race_releases_lock_and_only_then_waits(tmp_path):
    events=[]
    @contextmanager
    def lock(root):
        assert root==tmp_path
        events.append('locked')
        try:yield
        finally:events.append('released')
    calls=0
    def action():
        nonlocal calls
        calls+=1
        if calls==1:raise CapacityUnavailable('48 GPU cap')
        events.append('submitted')
        return 'job-handle'
    def sleep(seconds):
        assert seconds==30 and events[-1]=='released'
        events.append('waited')
    assert run_when_capacity(tmp_path,action,sleep=sleep,lock=lock)=='job-handle'
    assert events==['locked','released','waited','locked','submitted','released']


@pytest.mark.parametrize('error',[RuntimeError('ambiguous submission failure'),SystemExit('invalid configuration')])
def test_noncapacity_failure_is_never_retried(tmp_path,error):
    calls=0
    def action():
        nonlocal calls
        calls+=1
        raise error
    def forbidden_sleep(seconds):
        pytest.fail('must not wait/retry after an actual failure')
    with pytest.raises(type(error)):
        run_when_capacity(tmp_path,action,sleep=forbidden_sleep)
    assert calls==1
