import pytest
from decider_mlx.protocol import request_temperature

def test_original_type_temperature_for_isolated_score():
    cfg={'temperature':1.145,'temperature_by_type':{'choice':1.164,'noul':1.624,'score':1.124}}
    assert request_temperature(cfg,{'q':{'type':'score'}})==1.124
    assert request_temperature(cfg,{'q':{'type':'noul'}})==1.624
    assert request_temperature({'temperature':1.3},{'q':{'type':'choice'}})==1.3
    with pytest.raises(ValueError):request_temperature(cfg,{'a':{'type':'choice'},'b':{'type':'score'}})
