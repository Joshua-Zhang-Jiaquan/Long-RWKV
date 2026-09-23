from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qz'))
from confirm_capacity import PRIMARY,EXTRA,route_body


def body():
    return dict(framework_config=[dict(instance_count=1,spec_id='7166bd2e-6cbe-4bd9-be38-762d11003e7f',shm_gi=1800)])


def census(primary,extra):
    return dict(live_reserved_gpus=primary+extra,project_reserved_gpus={PRIMARY:primary,EXTRA:extra})


def test_preference_does_not_override_either_project_cap():
    b=body();assert route_body(b,census(24,8),preferred_project=PRIMARY)==PRIMARY
    b=body();assert route_body(b,census(32,8),preferred_project=PRIMARY)==EXTRA
    b=body();assert route_body(b,census(24,16))==PRIMARY
    with pytest.raises(SystemExit):route_body(body(),census(32,16),preferred_project=PRIMARY)


def test_unknown_project_and_unverified_resource_shape_rejected():
    with pytest.raises(ValueError):route_body(body(),census(0,0),preferred_project='unknown')
    b=body();b['framework_config'][0]['instance_count']=2
    with pytest.raises(ValueError):route_body(b,census(0,0),preferred_project=PRIMARY)
