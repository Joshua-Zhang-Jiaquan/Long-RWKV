import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('distance_shared',Path(__file__).resolve().parents[1]/'qz/distance_shared.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def test_execution_is_copied_outside_project_and_content_addressed(tmp_path):
    local=tmp_path/'project'/'execution.json';local.parent.mkdir();local.write_text('{"terminal_steps":600}\n')
    global_root=tmp_path/'global_user'
    shared=M.stage_execution(local,global_root)
    assert shared.is_relative_to(global_root)
    assert shared.read_bytes()==local.read_bytes()
    assert M.stage_execution(local,global_root)==shared
    local.write_text('{"terminal_steps":300}\n')
    new=M.stage_execution(local,global_root)
    assert new!=shared and shared.read_text()=='{"terminal_steps":600}\n'


def test_existing_shared_manifest_cannot_be_silently_overwritten(tmp_path):
    local=tmp_path/'execution.json';local.write_text('{}')
    shared=M.stage_execution(local,tmp_path/'global_user');shared.write_text('tampered')
    with pytest.raises(ValueError,match='conflict'):M.stage_execution(local,tmp_path/'global_user')
