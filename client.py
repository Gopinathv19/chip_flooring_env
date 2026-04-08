# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chip Flooring Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ChipFlooringAction, ChipFlooringObservation


class ChipFlooringEnv(
    EnvClient[ChipFlooringAction, ChipFlooringObservation, State]
):
    """
    Client for the Chip Flooring Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ChipFlooringEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(ChipFlooringAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ChipFlooringEnv.from_docker_image("chip_flooring_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ChipFlooringAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ChipFlooringAction) -> Dict:
        """
        Convert ChipFlooringAction to JSON payload for step message.

        Args:
            action: ChipFlooringAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ChipFlooringObservation]:
        """
        Parse server response into StepResult[ChipFlooringObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ChipFlooringObservation
        """
        obs_data = payload.get("observation", {})
        observation = ChipFlooringObservation(
            canva_space=obs_data.get("canva_space", [[0]]),
            remaining_blocks=obs_data.get("remaining_blocks", []),
            placed_blocks=obs_data.get("placed_blocks", []),
            block_summaries=obs_data.get("block_summaries", []),
            candidate_positions=obs_data.get("candidate_positions", []),
            density_map=obs_data.get("density_map", []),
            placement_focus=obs_data.get("placement_focus"),
            current_hpwl=obs_data.get("current_hpwl", 0.0),
            delta_hpwl=obs_data.get("delta_hpwl", 0.0),
            placed_block_count=obs_data.get("placed_block_count", 0),
            task_name=obs_data.get("task_name", "hard"),
            invalid_reasons=obs_data.get("invalid_reasons"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
