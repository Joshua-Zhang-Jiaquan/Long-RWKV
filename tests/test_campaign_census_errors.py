import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qz'))
import campaign as C


@pytest.mark.parametrize('payload',[
    {'ResponseMetadata':{'Error':{'Code':'AccessForbidden'}}},
    {'ResponseMetadata':{'Error':{'Code':'ServerError'}},'Result':{'jobs':[]}},
    {},{'Result':None},{'Result':{'jobs':'invalid'}},
])
def test_zero_exit_api_failure_does_not_mean_free_capacity(monkeypatch,payload):
    monkeypatch.setattr(C.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps(payload),stderr=''))
    with pytest.raises(SystemExit):C.list_all_jobs()


def test_valid_pagination_collects_all_jobs(monkeypatch):
    payloads=iter([{'Result':{'jobs':[{'job_id':'one'}]}},{'Result':{'jobs':[{'job_id':'two'}]}},{'Result':{'jobs':[]}}])
    monkeypatch.setattr(C.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps(next(payloads)),stderr=''))
    assert C.list_all_jobs()==[{'job_id':'one'},{'job_id':'two'}]


def test_truncated_pagination_refuses_submission_census(monkeypatch):
    monkeypatch.setattr(C.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps({'Result':{'jobs':[{'job_id':'one'}]}}),stderr=''))
    with pytest.raises(SystemExit,match='pagination limit'):C.list_all_jobs()
