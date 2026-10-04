import pytest
from dating_compat.data import DataError
from dating_compat.final_evaluation import final_evaluation


def test_existing_selection_required_before_evaluation(tmp_path,monkeypatch):
    from dating_compat import final_evaluation as module
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module,'load_records',lambda *args:pytest.fail('Must gate before data access'))
    with pytest.raises(DataError,match='Existing frozen selection'):
        final_evaluation()
