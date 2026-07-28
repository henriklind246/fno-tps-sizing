from __future__ import annotations

from dataclasses import replace

import pytest

from fno_tps.config import MeshConfig, TimeConfig, load_study_config
from fno_tps.problem import HeatingEvent, SimulationCase


@pytest.fixture
def demo_config():
    config = load_study_config("conf/demo.yaml")
    return replace(
        config,
        mesh=MeshConfig(nx_tps=4, nx_bond=1, nx_back=3, ny=8),
        time=TimeConfig(dt=1.0, t_final=4.0, save_stride=1, horizon_candidates=(4.0,)),
        training=replace(
            config.training,
            batch_size=2,
            epochs=1,
            patience=1,
            targets_per_case=4,
            modes1=2,
            modes2=2,
            width=8,
            layers=2,
            cond_hidden=16,
            temporal_hidden=8,
            forcing_embed_dim=4,
        ),
    )


@pytest.fixture
def incident_config():
    config = load_study_config("conf/incident-radiation-demo.yaml")
    return replace(
        config,
        mesh=MeshConfig(nx_tps=4, nx_bond=1, nx_back=3, ny=8),
        time=TimeConfig(
            dt=0.25,
            t_final=4.0,
            save_stride=1,
            horizon_candidates=(4.0,),
        ),
    )


@pytest.fixture
def heating_event():
    return HeatingEvent(
        amplitude=10000.0,
        y_center=0.10,
        t_center=1.5,
        sigma_y=0.025,
        sigma_t=0.5,
    )


@pytest.fixture
def three_cases(heating_event):
    return [
        SimulationCase(
            case_id=f"case-{index}",
            d_tps=0.02,
            heating_events=(heating_event,),
            bond_defects=(),
        )
        for index in range(3)
    ]
