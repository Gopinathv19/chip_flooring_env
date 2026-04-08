# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Chip Flooring Env Environment.

The chip_flooring_env environment is a simple test environment that echoes back messages.
"""

from __future__ import annotations

from typing import Any,Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ChipFlooringAction(Action):
    """ 
    Action for the chip flooring environment 
    """

    x : int = Field(default=0,description="Used to identify the x coordinate")
    y : int = Field(default=0,description="Used to identify the y coordinate")
    choosen_block_index : int = Field(default=0 ,description="The agent which picks the block from the remaining blocks")


 


class ChipFlooringObservation(Observation):
    """Observation from the Chip Flooring Env environment """

    canva_space : list[list[int]] = Field(default=[[0]],description="The grid type structure to represent the canva space")
    remaining_blocks : list[Any] = Field(default_factory=list,description="Used to give the agent detailing abouth what are all the remaining block are there")
    placed_blocks : list[Any] = Field(default_factory=list,description="Used to give the agent so far placed blocks")
    block_summaries : list[Any] = Field(default_factory=list,description="Connectivity summary for remaining blocks")
    candidate_positions : list[Any] = Field(default_factory=list,description="Small scored set of legal placement candidates")
    density_map : list[list[float]] = Field(default_factory=list,description="Coarse occupancy map for congestion awareness")
    placement_focus : Optional[dict[str, Any]] = Field(default=None,description="Most important block to place next")
    current_hpwl : float = Field(default=0.0,description="Current total wirelength cost for all fully placed connections")
    delta_hpwl : float = Field(default=0.0,description="Wirelength cost added by the latest placement")
    placed_block_count : int = Field(default=0,description="Number of blocks already placed")
    task_name : str = Field(default="hard",description="Current task difficulty name")
    invalid_reasons:Optional[str] = Field(default=None,description="Reason for the last action was rejected")



class ChipFlooringResponseState(State):
    "State for the Chip Flooring Environment to track the changes in the environment"

    episode_id : str = Field(default="",description="Used to identify the episode id")
    step_count : int = Field(default=0,description="Used to identify the step count")
    grid_size : int = Field(default=24,description="Used to identify the grid size")
    grid : list[list[int]] = Field(default=[[0]],description="Used to identify the grid")     
    blocks : list[Any] = Field(default_factory=list,description="Used to identify the blocks")
    placed_blocks : list[Any] = Field(default_factory=list,description="Used to identify the placed blocks")
    remaining_blocks : list[Any] = Field(default_factory=list,description="Used to identify the remaining blocks")
    done : bool = Field(default=False,description="Used to identify if the episode is done")
    reward : float = Field(default=0.0,description="This is used to update the model thinking and the trajectory")
    current_hpwl : float = Field(default=0.0,description="Current total HPWL score for the placed layout")
    delta_hpwl : float = Field(default=0.0,description="HPWL cost added by the latest placement")
    task_name : str = Field(default="hard",description="Current difficulty task name")
    trajectory: list[Any] = Field(default_factory=list,description="used to map the entire trajectory of the agent decission")
    
    
