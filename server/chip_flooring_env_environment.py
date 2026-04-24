import copy
from uuid import uuid4
import os
import random
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ChipFlooringAction, ChipFlooringObservation, ChipFlooringResponseState
except ImportError:
    from models import ChipFlooringAction, ChipFlooringObservation, ChipFlooringResponseState




class ChipFlooringEnvironment(Environment):
 
    def __init__(self):
        """Initialize the chip_flooring_env environment."""
        self.task_name = os.getenv("TASK_NAME", "hard_standard_long_horizon").strip().lower() or "hard_standard_long_horizon"
        self.grid_size = 24
        self.hpwl_weight = 0.25
        self.valid_placement_bonus = 0.05
        self.final_completion_bonus = 0.75
        self.invalid_block_penalty = -0.8
        self.invalid_overlap_penalty = -0.6
        self.invalid_bounds_penalty = -0.5
        self._reset_count = 0   
        self.canvas = None
        self.blocks = []
        self.block_id_map = {}
        self._block_lookup = {}
        self.phase = "placement"
        self.phase_step = 0
        self.instruction = ""
        self._long_horizon_hidden_edges: set[tuple[str, str]] = set()
        self._long_horizon_constraints_revealed = False
        self._long_horizon_phase_boundaries = {
            "reveal": 0,
            "repair": 0,
            "finalize": 0,
        }
        self._long_horizon_recent_positions: dict[str, list[tuple[int, int]]] = {}
        self._long_horizon_move_counts: dict[str, int] = {}
        self.task_aliases = self._build_task_aliases()
        self.task_configs = self._build_task_configs()
        self.task_name = self._resolve_task_name(self.task_name)
        self.global_netlist = self._select_task_netlist(self.task_name)
        self.grid_size = self._select_task_grid_size(self.task_name)
        self._state = ChipFlooringResponseState(
            episode_id=str(uuid4()),
            step_count=0,
            grid_size=self.grid_size,
            grid=[[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)],
            blocks=[],
            placed_blocks=[],
            remaining_blocks=[],
            done=False,
            reward=0.0,
            current_hpwl=0.0,
            delta_hpwl=0.0,
            task_name=self.task_name,
            phase=self.phase,
            phase_step=self.phase_step,
            instruction=self.instruction,
            hidden_constraint_count=0,
            revealed_constraint_count=0,
            trajectory=[],
        )
 


    def reset(self) -> ChipFlooringObservation:
        """
        Reset the environment.

        Returns:
            ChipFlooringObservation with a ready message
        """
        self._reset_count += 1
        self.canvas = Canvas(self.grid_size)
        self.blocks = self._convert_global_netlist_to_blocks()
        self._block_lookup = {block.id: block for block in self.blocks}
        self.block_id_map = {
            block.id: idx + 1 for idx, block in enumerate(self.blocks)
        }

        self._place_fixed_blocks()
        self._configure_long_horizon_episode()

        self._state = ChipFlooringResponseState(
            episode_id=str(uuid4()),
            step_count=0,
            grid_size=self.grid_size,
            grid=self.canvas.grid,
            blocks=self.blocks,
            placed_blocks=[block for block in self.blocks if block.fixed],
            remaining_blocks=[block for block in self.blocks if not block.fixed],
            done=False,
            reward=0,
            current_hpwl=0.0,
            delta_hpwl=0.0,
            task_name=self.task_name,
            phase=self.phase,
            phase_step=self.phase_step,
            instruction=self.instruction,
            hidden_constraint_count=len(self._long_horizon_hidden_edges),
            revealed_constraint_count=0,
        )

        return self._build_observation()

         
    def step(self, action: ChipFlooringAction) -> ChipFlooringObservation:  # type: ignore[override]
        if self.canvas is None:
            self.reset()
 
        self._state.step_count += 1
        self._state.reward=0.0
        self._state.done=False
        self._state.delta_hpwl = 0.0
        self._update_phase()
        
        invalid_reasons = None
        row = action.x
        col = action.y
        current_block_index = action.choosen_block_index
        action_mode = str(getattr(action, "action_type", "place") or "place").strip().lower()
        effective_action_mode = action_mode

        if not isinstance(current_block_index,int) or current_block_index < 0 or current_block_index >= len(self._state.blocks):
            invalid_reasons = "Invalid Block Index correclty choose the block index with in the range"
            self._state.reward=self.invalid_block_penalty
        else:
            block = self._state.blocks[current_block_index]
            can_move = (
                self._is_long_horizon_task()
                and self.phase in {"repair", "finalize"}
                and block in self._state.placed_blocks
                and not block.fixed
                and not getattr(block, "committed", False)
            )
            can_commit = (
                self._is_long_horizon_task()
                and self.phase in {"repair", "finalize"}
                and block in self._state.placed_blocks
                and not block.fixed
                and action_mode == "commit"
            )

            if can_commit:
                effective_action_mode = "commit"
                block.fixed = True
                block.committed = True
                self._long_horizon_recent_positions.pop(block.id, None)
                self._long_horizon_move_counts.pop(block.id, None)
                self._state.reward = 0.2 if self.phase == "finalize" else 0.1
            elif can_move:
                effective_action_mode = "move"
                if block.position == (row, col):
                    invalid_reasons = "The selected move keeps the block in the same position"
                    self._state.reward = self.invalid_block_penalty
                elif (row, col) in self._long_horizon_recent_positions.get(block.id, []):
                    invalid_reasons = "The selected move repeats a recently visited position"
                    self._state.reward = self.invalid_block_penalty
                else:
                    previous_hpwl = self._state.current_hpwl if self._state.current_hpwl > 0 else self._compute_total_hpwl()
                    if not self._move_block(block, (row, col)):
                        invalid_reasons = "The given possition cannot be occupied after moving the selected block"
                        self._state.reward = self.invalid_bounds_penalty
                    else:
                        recent_positions = self._long_horizon_recent_positions.setdefault(block.id, [])
                        recent_positions.append((row, col))
                        self._long_horizon_recent_positions[block.id] = recent_positions[-8:]
                        self._long_horizon_move_counts[block.id] = self._long_horizon_move_counts.get(block.id, 0) + 1
                        block.move_count = self._long_horizon_move_counts[block.id]
                        incremental_hpwl = self._compute_total_hpwl() - previous_hpwl
                        self._state.delta_hpwl = incremental_hpwl
                        self._state.current_hpwl = previous_hpwl + incremental_hpwl
                        placed_neighbor_weight = self._placed_neighbor_weight(block)
                        move_penalty = 0.15 * max(1.0, float(block.power))
                        oscillation_penalty = 0.04 * float(self._long_horizon_move_counts.get(block.id, 0))
                        phase_scale = 0.6 if self.phase == "repair" else 0.9
                        self._state.reward = (
                            phase_scale * (self.valid_placement_bonus + (0.02 * placed_neighbor_weight) - (self.hpwl_weight * incremental_hpwl * (max(1.0, float(block.power)) ** 2)))
                            - move_penalty
                            - oscillation_penalty
                        )
            elif block in self._state.remaining_blocks:
                effective_action_mode = "place"
                if not self.canvas.can_occupy((row, col), block.y, block.x):
                    invalid_reasons = "The given possition cannot be occupied check the canvas once again for the right placment"
                    self._state.reward = self.invalid_bounds_penalty
                else:
                    block_num = self.block_id_map[block.id]
                    self.canvas.occupy_region((row, col), block.y, block.x, block_num)
                    block.placed=True
                    block.position=(row, col)
                    self._state.placed_blocks.append(block)
                    self._state.remaining_blocks.remove(block)
                    incremental_hpwl = self._compute_incremental_hpwl(block)
                    placed_neighbor_weight = self._placed_neighbor_weight(block)
                    self._state.delta_hpwl = incremental_hpwl
                    self._state.current_hpwl = self._compute_total_hpwl()
                    power_scale = max(1.0, float(block.power))
                    distance_penalty = 0.0
                    if block.type == "macro":
                        center = self._block_center(block)
                        if center is not None:
                            grid_center = (self.grid_size / 2.0, self.grid_size / 2.0)
                            distance_penalty = 0.05 * power_scale * self._manhattan_distance(center, grid_center) / self.grid_size
                    phase_scale = 0.25 if self._is_long_horizon_task() and self.phase == "placement" else 1.0
                    self._state.reward = phase_scale * (
                        self.valid_placement_bonus
                        + (0.02 * placed_neighbor_weight)
                        - (self.hpwl_weight * incremental_hpwl * (power_scale ** 2))
                        - distance_penalty
                    )
            else:
                invalid_reasons = "The selected block cannot be acted on in the current long horizon phase"
                self._state.reward = self.invalid_block_penalty

            if self._is_long_horizon_task():
                all_movable_committed = all(
                    block.fixed or getattr(block, "committed", False)
                    for block in self._state.placed_blocks
                ) and len(self._state.remaining_blocks) == 0
                finalize_step = self._long_horizon_phase_boundaries["finalize"]
                if (self._state.step_count >= finalize_step and len(self._state.remaining_blocks) == 0) or all_movable_committed:
                    self._state.done = True
                else:
                    self._state.done = False
            else:
                self._state.done = len(self._state.remaining_blocks) == 0

            if self._state.done:
                final_bonus = 2.0 if self._is_long_horizon_task() else self.final_completion_bonus
                self._state.reward += final_bonus - (self.hpwl_weight * self._state.current_hpwl)
        
        self._state.grid = self.canvas.grid
        self._state.phase = self.phase
        self._state.phase_step = self.phase_step
        self._state.instruction = self.instruction
        self._state.hidden_constraint_count = len(self._long_horizon_hidden_edges) if self._is_long_horizon_task() else 0
        self._state.revealed_constraint_count = len(self._long_horizon_hidden_edges) if self._is_long_horizon_task() and self._long_horizon_constraints_revealed else 0
        self._state.trajectory.append(
            {
                "step_count": self._state.step_count,
                "action": {
                    "action_type": effective_action_mode,
                    "x": row,
                    "y": col,
                    "choosen_block_index": current_block_index,
                },
                "reward": self._state.reward,
                "done": self._state.done,
                "invalid_reason": invalid_reasons,
                "current_hpwl": self._state.current_hpwl,
                "delta_hpwl": self._state.delta_hpwl,
                "task_name": self._state.task_name,
                "phase": self._state.phase,
                "phase_step": self._state.phase_step,
                "remaining_blocks": [b.id for b in self._state.remaining_blocks],
                "placed_blocks": [b.id for b in self._state.placed_blocks],
            }
        )

        return self._build_observation(invalid_reason=invalid_reasons)
            


        
 

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

 

    '''
    This Function is used to convert the global netlist to blocks
    '''

    def _convert_global_netlist_to_blocks(self):
        nodes = self.global_netlist["nodes"]
        edges = self.global_netlist["edges"]

        blocks = {}

        for node in nodes:
            block = Block(
                id=node["id"],
                height=node["height"],
                width=node["width"],
                block_type=node.get("type", "standard"),
                power=float(node.get("power", 1.0)),
                fixed=bool(node.get("fixed", False)),
            )
            if node.get("position") is not None:
                block.position = tuple(node["position"])
            blocks[node["id"]] = block
        
        for edge in edges:
            src = edge["from"]
            dist = edge["to"]
            weight = edge["weight"]
            criticality = float(edge.get("criticality", 0.0))
            edge_key = tuple(sorted((str(src), str(dist))))
            blocks[src].connect_block(blocks[dist], weight, criticality, edge_key=edge_key)
            blocks[dist].connect_block(blocks[src], weight, criticality, edge_key=edge_key)

        return list(blocks.values())

    def _place_fixed_blocks(self) -> None:
        if self.canvas is None:
            return

        for block in self.blocks:
            if not block.fixed:
                continue
            if block.position is None:
                raise ValueError(f"Fixed block {block.id} is missing a position")
            block_num = self.block_id_map[block.id]
            row, col = block.position
            if not self.canvas.can_occupy((row, col), block.y, block.x):
                raise ValueError(f"Fixed block {block.id} cannot be placed at {block.position}")
            self.canvas.occupy_region((row, col), block.y, block.x, block_num)
            block.placed = True

    def _is_long_horizon_task(self) -> bool:
        return str(self.task_name).strip().lower().endswith("_long_horizon") or self.task_name == "long_horizon"

    def _edge_key(self, edge_info: dict, default: tuple[str, str] | None = None) -> tuple[str, str] | None:
        key = edge_info.get("edge_key")
        if isinstance(key, tuple) and len(key) == 2:
            return (str(key[0]), str(key[1]))
        return default

    def _should_reveal_hidden_constraints(self) -> bool:
        return self._is_long_horizon_task() and self._state.step_count >= self._long_horizon_phase_boundaries["reveal"]

    def _build_task_aliases(self) -> dict[str, str]:
        return {
            "easy": "easy_standard_long_horizon",
            "medium": "medium_standard_long_horizon",
            "hard": "hard_standard_long_horizon",
            "heterogeneous": "hard_heterogeneous_long_horizon",
            "fixed_obstacles": "hard_fixed_obstacles_long_horizon",
            "long_horizon": "hard_heterogeneous_long_horizon",
        }

    def _resolve_task_name(self, task_name: str) -> str:
        normalized = str(task_name or "").strip().lower().replace("-", "_")
        if not normalized:
            return "hard_standard_long_horizon"
        if normalized in self.task_configs:
            return normalized
        if normalized in self.task_aliases:
            return self.task_aliases[normalized]
        if normalized.endswith("_long_horizon") and normalized in self.task_configs:
            return normalized
        return "hard_standard_long_horizon"

    def _clone_task_config(self, config: dict) -> dict:
        return {
            key: copy.deepcopy(value) if key in {"nodes", "edges"} else copy.deepcopy(value)
            for key, value in config.items()
        }

    def _apply_heterogeneous_scenario(self, config: dict, difficulty: str) -> dict:
        macro_count_by_difficulty = {"easy": 1, "medium": 3, "hard": 5}
        macro_count = macro_count_by_difficulty.get(difficulty, 2)
        nodes = config.get("nodes", [])
        edges = config.get("edges", [])
        degree: dict[str, float] = {str(node["id"]): 0.0 for node in nodes}
        for edge in edges:
            src = str(edge.get("from"))
            dst = str(edge.get("to"))
            weight = float(edge.get("weight", 0.0))
            degree[src] = degree.get(src, 0.0) + weight
            degree[dst] = degree.get(dst, 0.0) + weight

        ranked_nodes = sorted(nodes, key=lambda node: (-degree.get(str(node["id"]), 0.0), str(node["id"])))
        for index, node in enumerate(ranked_nodes[:macro_count]):
            node["type"] = "macro"
            node["power"] = round(2.4 + (0.35 * index) + (0.15 * macro_count), 2)

        return config

    def _apply_fixed_obstacles_scenario(self, config: dict, difficulty: str) -> dict:
        grid_size = int(config.get("grid_size", self.grid_size))
        obstacle_specs_by_difficulty = {
            "easy": [
                {"id": "P", "height": 2, "width": 2, "fixed": True, "position": [2, 2]},
                {"id": "Q", "height": 1, "width": 3, "fixed": True, "position": [max(0, grid_size - 3), 2]},
            ],
            "medium": [
                {"id": "P", "height": 2, "width": 2, "fixed": True, "position": [grid_size // 3, grid_size // 3]},
                {"id": "Q", "height": 1, "width": 4, "fixed": True, "position": [max(0, grid_size - 4), 3]},
                {"id": "R", "height": 3, "width": 1, "fixed": True, "position": [3, max(0, grid_size - 4)]},
            ],
            "hard": [
                {"id": "P", "height": 2, "width": 2, "fixed": True, "position": [8, 8]},
                {"id": "Q", "height": 1, "width": 4, "fixed": True, "position": [15, 4]},
                {"id": "R", "height": 3, "width": 1, "fixed": True, "position": [4, 16]},
            ],
        }
        obstacle_nodes = obstacle_specs_by_difficulty.get(difficulty, obstacle_specs_by_difficulty["medium"])
        config["nodes"].extend(copy.deepcopy(obstacle_nodes))
        anchor_nodes = [str(node["id"]) for node in config["nodes"] if not bool(node.get("fixed"))][:3]
        obstacle_ids = [str(node["id"]) for node in obstacle_nodes]
        for obstacle_id, anchor_id in zip(obstacle_ids, anchor_nodes):
            config["edges"].append(
                {
                    "from": obstacle_id,
                    "to": anchor_id,
                    "weight": 1.2 + 0.2 * len(anchor_id),
                    "criticality": 0.7,
                }
            )
        return config

    def _apply_long_horizon_profile(self, config: dict, difficulty: str, scenario: str) -> dict:
        total_blocks = len(config.get("nodes", []))
        scenario_bias = {
            "standard": 0,
            "heterogeneous": 2,
            "fixed_obstacles": 3,
        }.get(scenario, 1)
        difficulty_bias = {"easy": 0, "medium": 2, "hard": 4}.get(difficulty, 2)
        total_steps = max(24, total_blocks + 8 + scenario_bias + difficulty_bias)
        reveal_step = max(4, total_blocks // 3)
        repair_step = max(reveal_step + 4, (2 * total_steps) // 3)
        finalize_step = max(repair_step + 1, total_steps)
        hidden_fraction = {
            "standard": 0.25,
            "heterogeneous": 0.35,
            "fixed_obstacles": 0.4,
        }.get(scenario, 0.3)
        if difficulty == "easy":
            hidden_fraction -= 0.05
        elif difficulty == "hard":
            hidden_fraction += 0.05

        config["phase_reveal_step"] = reveal_step
        config["phase_repair_step"] = repair_step
        config["phase_finalize_step"] = finalize_step
        config["hidden_edge_fraction"] = round(max(0.15, min(0.55, hidden_fraction)), 2)
        config["difficulty"] = difficulty
        config["scenario"] = scenario
        config["horizon"] = "long_horizon"
        return config

    def _build_composite_task_configs(self, base_configs: dict[str, dict]) -> dict[str, dict]:
        composite_configs: dict[str, dict] = {}
        for difficulty in ("easy", "medium", "hard"):
            standard = self._apply_long_horizon_profile(
                self._clone_task_config(base_configs[difficulty]),
                difficulty,
                "standard",
            )
            composite_configs[f"{difficulty}_standard_long_horizon"] = standard

            heterogeneous = self._apply_heterogeneous_scenario(
                self._clone_task_config(base_configs[difficulty]),
                difficulty,
            )
            composite_configs[f"{difficulty}_heterogeneous_long_horizon"] = self._apply_long_horizon_profile(
                heterogeneous,
                difficulty,
                "heterogeneous",
            )

            fixed_obstacles = self._apply_fixed_obstacles_scenario(
                self._clone_task_config(base_configs[difficulty]),
                difficulty,
            )
            composite_configs[f"{difficulty}_fixed_obstacles_long_horizon"] = self._apply_long_horizon_profile(
                fixed_obstacles,
                difficulty,
                "fixed_obstacles",
            )

        return composite_configs

    def _configure_long_horizon_episode(self) -> None:
        self.phase = "placement"
        self.phase_step = 0
        self.instruction = ""
        self._long_horizon_hidden_edges = set()
        self._long_horizon_constraints_revealed = False
        self._long_horizon_phase_boundaries = {
            "reveal": 0,
            "repair": 0,
            "finalize": 0,
        }
        self._long_horizon_recent_positions = {}
        self._long_horizon_move_counts = {}

        if not self._is_long_horizon_task():
            return

        config = self.task_configs.get(self.task_name, {})
        candidate_edges: list[tuple[tuple[str, str], float]] = []
        for edge in self.global_netlist["edges"]:
            key = tuple(sorted((str(edge["from"]), str(edge["to"]))))
            criticality = float(edge.get("criticality", 0.0))
            if criticality >= 0.55:
                candidate_edges.append((key, criticality))

        candidate_edges.sort(key=lambda item: (-item[1], item[0]))
        hidden_fraction = float(config.get("hidden_edge_fraction", 0.35))
        hidden_count = max(3, int(round(len(candidate_edges) * hidden_fraction)))
        self._long_horizon_hidden_edges = {key for key, _ in candidate_edges[:hidden_count]}

        total_steps = int(config.get("phase_finalize_step", max(24, len(self.blocks) + 8)))
        reveal_step = int(config.get("phase_reveal_step", max(4, len(self.blocks) // 3)))
        repair_step = int(config.get("phase_repair_step", max(reveal_step + 4, (2 * total_steps) // 3)))
        finalize_step = max(repair_step + 1, total_steps)
        self._long_horizon_phase_boundaries = {
            "reveal": reveal_step,
            "repair": repair_step,
            "finalize": finalize_step,
        }
        self.phase = "placement"
        self.phase_step = 0
        self.instruction = "Establish the coarse layout using the visible constraints."

    def _update_phase(self) -> None:
        if not self._is_long_horizon_task():
            self.phase = "placement"
            self.phase_step = self._state.step_count
            self.instruction = ""
            return

        boundaries = self._long_horizon_phase_boundaries
        if self._state.step_count < boundaries["reveal"]:
            self.phase = "placement"
            self.phase_step = self._state.step_count
            self.instruction = "Establish the coarse layout using the visible constraints."
            return

        if not self._long_horizon_constraints_revealed:
            self._long_horizon_constraints_revealed = True

        if self._state.step_count < boundaries["repair"]:
            self.phase = "repair"
            self.phase_step = self._state.step_count - boundaries["reveal"]
            self.instruction = "Repair the coarse layout after the hidden constraints are revealed."
            return

        self.phase = "finalize"
        self.phase_step = self._state.step_count - boundaries["repair"]
        self.instruction = "Stabilize the final layout and finish with the best remaining placements."

    def _is_hidden_edge(self, edge_info: dict) -> bool:
        key = self._edge_key(edge_info)
        return key in self._long_horizon_hidden_edges if key is not None else False

    def _display_edge_importance(self, edge_info: dict) -> float:
        importance = self._edge_importance(edge_info)
        if self._is_long_horizon_task() and not self._long_horizon_constraints_revealed and self._is_hidden_edge(edge_info):
            return max(0.1, importance * 0.2)
        return importance

    def _move_block(self, block: "Block", new_position: tuple[int, int]) -> bool:
        if self.canvas is None or block.position is None:
            return False

        old_position = block.position
        if new_position == old_position:
            return False

        self.canvas.remove_region(old_position, block.y, block.x)
        if not self.canvas.can_occupy(new_position, block.y, block.x):
            self.canvas.occupy_region(old_position, block.y, block.x, self.block_id_map[block.id])
            return False

        self.canvas.occupy_region(new_position, block.y, block.x, self.block_id_map[block.id])
        block.position = new_position
        return True

    def _edge_importance(self, edge_info: dict) -> float:
        weight = float(edge_info.get("weight", 0.0))
        criticality = float(edge_info.get("criticality", 0.0))
        return weight * (1.0 + criticality)

    def _block_center(self, block: "Block"):
        if block.position is None:
            return None
        row, col = block.position
        return (row + (block.x / 2.0), col + (block.y / 2.0))

    def _manhattan_distance(self, point_a, point_b):
        return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])

    def _compute_incremental_hpwl(self, placed_block: "Block") -> float:
        """
        Compute the wirelength added by the latest placement.

        Only connections whose other endpoint is already placed are counted here,
        so each edge is charged once when its second endpoint lands.
        """
        placed_center = self._block_center(placed_block)
        if placed_center is None:
            return 0.0

        total = 0.0
        for neighbor_id, edge_info in placed_block.get_internal_netlist().items():
            neighbor = self._block_lookup.get(neighbor_id)
            if neighbor is None or not neighbor.placed:
                continue
            neighbor_center = self._block_center(neighbor)
            if neighbor_center is None:
                continue
            total += self._edge_importance(edge_info) * self._manhattan_distance(placed_center, neighbor_center) / self.grid_size

        return total

    def _placed_neighbor_weight(self, placed_block: "Block") -> float:
        total = 0.0
        for neighbor_id, edge_info in placed_block.get_internal_netlist().items():
            neighbor = self._block_lookup.get(neighbor_id)
            if neighbor is not None and neighbor.placed:
                total += self._edge_importance(edge_info)
        return total

    def _compute_total_hpwl(self) -> float:
        """
        Compute the total weighted HPWL for the currently placed layout.

        This only counts edges where both endpoints are already placed.
        """
        total = 0.0
        for edge in self.global_netlist["edges"]:
            src = self._block_lookup.get(edge["from"])
            dst = self._block_lookup.get(edge["to"])
            if src is None or dst is None or not src.placed or not dst.placed:
                continue
            src_center = self._block_center(src)
            dst_center = self._block_center(dst)
            if src_center is None or dst_center is None:
                continue
            total += self._edge_importance(edge) * self._manhattan_distance(src_center, dst_center) / self.grid_size

        return total

    def _block_priority_score(self, block: "Block") -> float:
        placed_neighbor_weight = 0.0
        placed_neighbor_count = 0
        for neighbor_id, edge_info in block.get_internal_netlist().items():
            neighbor = self._block_lookup.get(neighbor_id)
            if neighbor is not None and neighbor.placed:
                placed_neighbor_weight += self._display_edge_importance(edge_info)
                placed_neighbor_count += 1
        degree = len(block.get_internal_netlist())
        area = block.x * block.y
        return (
            (placed_neighbor_weight * 10.0)
            + (placed_neighbor_count * 2.0)
            + (degree * 0.25)
            + (area * 0.05)
            + (block.power * 5.0)
        )

    def _rank_remaining_blocks(self) -> list["Block"]:
        return sorted(
            self._state.remaining_blocks,
            key=lambda block: (
                -self._block_priority_score(block),
                -(block.x * block.y),
                block.id,
            ),
        )

    def _block_summary(self, block: "Block") -> dict:
        neighbors = []
        placed_neighbors = []
        unplaced_neighbors = []
        strongest_neighbors = []

        for neighbor_id, edge_info in sorted(
            block.get_internal_netlist().items(),
            key=lambda item: (-self._display_edge_importance(item[1]), item[0]),
        ):
            neighbor = self._block_lookup.get(neighbor_id)
            importance = self._display_edge_importance(edge_info)
            criticality = float(edge_info.get("criticality", 0.0))
            if self._is_long_horizon_task() and not self._long_horizon_constraints_revealed and self._is_hidden_edge(edge_info):
                criticality = round(max(0.05, criticality * 0.2), 4)
            neighbor_summary = {
                "id": neighbor_id,
                "weight": edge_info.get("weight", 0.0),
                "criticality": criticality,
                "importance": round(importance, 4),
                "placed": bool(neighbor.placed) if neighbor is not None else False,
                "position": neighbor.position if neighbor is not None else None,
            }
            neighbors.append(neighbor_summary)
            if neighbor is not None and neighbor.placed:
                placed_neighbors.append(neighbor_summary)
            else:
                unplaced_neighbors.append(neighbor_summary)
            if len(strongest_neighbors) < 3:
                strongest_neighbors.append(neighbor_summary)

        return {
            "id": block.id,
            "height": block.x,
            "width": block.y,
            "type": block.type,
            "power": block.power,
            "fixed": block.fixed,
            "committed": bool(getattr(block, "committed", False)),
            "move_count": int(self._long_horizon_move_counts.get(block.id, 0)),
            "area": block.x * block.y,
            "degree": len(block.get_internal_netlist()),
            "priority_score": self._block_priority_score(block),
            "connected_blocks": [neighbor["id"] for neighbor in neighbors],
            "strongest_neighbors": strongest_neighbors,
            "placed_neighbors": placed_neighbors,
            "unplaced_neighbors": unplaced_neighbors,
            "placed_neighbor_count": len(placed_neighbors),
            "placed_neighbor_weight": sum(n["importance"] for n in placed_neighbors),
            "placed": block.placed,
            "position": block.position,
        }

    def _coarse_density_map(self, cells: int = 6) -> list[list[float]]:
        density = [[0.0 for _ in range(cells)] for _ in range(cells)]
        if self.canvas is None or self.grid_size <= 0:
            return density

        cell_h = max(1, self.grid_size // cells)
        cell_w = max(1, self.grid_size // cells)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.canvas.grid[row][col] == 0:
                    continue
                r = min(cells - 1, row // cell_h)
                c = min(cells - 1, col // cell_w)
                density[r][c] += 1.0

        cell_area = float(cell_h * cell_w)
        if cell_area <= 0:
            return density

        for r in range(cells):
            for c in range(cells):
                density[r][c] = round(density[r][c] / cell_area, 3)

        return density

    def _anchor_score(self, block: "Block", row: int, col: int) -> float:
        center = (row + (block.x / 2.0), col + (block.y / 2.0))
        score = 0.0
        placed_neighbor_count = 0
        for neighbor_id, edge_info in block.get_internal_netlist().items():
            neighbor = self._block_lookup.get(neighbor_id)
            if neighbor is None or not neighbor.placed:
                continue
            neighbor_center = self._block_center(neighbor)
            if neighbor_center is None:
                continue
            score -= self._display_edge_importance(edge_info) * self._manhattan_distance(center, neighbor_center) / self.grid_size
            placed_neighbor_count += 1

        if placed_neighbor_count == 0:
            grid_center = (self.grid_size / 2.0, self.grid_size / 2.0)
            score -= 0.05 * block.power * self._manhattan_distance(center, grid_center) / self.grid_size

        density = self._coarse_density_map()
        density_row = min(len(density) - 1, max(0, row * len(density) // self.grid_size))
        density_col = min(len(density[0]) - 1, max(0, col * len(density[0]) // self.grid_size))
        score -= 0.15 * block.power * density[density_row][density_col]

        score *= (1.0 + (0.3 * block.power))
        return round(score, 4)

    def _generate_candidate_positions(
        self,
        top_blocks: int = 4,
        per_block_limit: int = 3,
    ) -> list[dict]:
        if self.canvas is None:
            return []

        candidates: list[dict] = []
        ranked_blocks = self._rank_remaining_blocks()[:top_blocks]

        for block in ranked_blocks:
            block_candidates: list[dict] = []
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if not self.canvas.can_occupy((row, col), block.y, block.x):
                        continue
                    block_candidates.append(
                        {
                            "block_id": block.id,
                            "x": row,
                            "y": col,
                            "height": block.x,
                            "width": block.y,
                            "score": self._anchor_score(block, row, col),
                            "congestion_score": self._cluster_congestion_score(block,row,col),
                            "priority_score": round(self._block_priority_score(block), 4),
                        }
                    )
            block_candidates.sort(key=lambda item: (-item["score"], item["x"], item["y"]))
            candidates.extend(block_candidates[:per_block_limit])

        candidates.sort(key=lambda item: (-item["priority_score"], -item["score"], item["block_id"], item["x"], item["y"]))

        if self._is_long_horizon_task() and self.phase in {"repair", "finalize"}:
            movable_placed_blocks = [
                block for block in self._state.placed_blocks
                if not block.fixed and block.position is not None and not getattr(block, "committed", False)
            ]
            movable_placed_blocks.sort(
                key=lambda block: (
                    -self._block_priority_score(block),
                    -(block.power),
                    block.id,
                )
            )
            for block in movable_placed_blocks[:top_blocks]:
                original_position = block.position
                block_num = self.block_id_map[block.id]
                self.canvas.remove_region(original_position, block.y, block.x)
                repair_candidates: list[dict] = []
                move_count = self._long_horizon_move_counts.get(block.id, 0)
                if self.phase == "finalize" or move_count > 0:
                    repair_candidates.append(
                        {
                            "block_id": block.id,
                            "x": int(original_position[0]),
                            "y": int(original_position[1]),
                            "height": block.x,
                            "width": block.y,
                            "score": round(
                                self._anchor_score(block, int(original_position[0]), int(original_position[1]))
                                + (0.15 if self.phase == "repair" else 0.5)
                                + (0.1 * max(0, 2 - move_count)),
                                4,
                            ),
                            "congestion_score": self._cluster_congestion_score(block, int(original_position[0]), int(original_position[1])),
                            "priority_score": round(self._block_priority_score(block), 4),
                            "action_type": "commit",
                        }
                    )
                for row in range(self.grid_size):
                    for col in range(self.grid_size):
                        if (row, col) == original_position:
                            continue
                        if (row, col) in self._long_horizon_recent_positions.get(block.id, []):
                            continue
                        if self.phase == "finalize" and move_count >= 2:
                            continue
                        if not self.canvas.can_occupy((row, col), block.y, block.x):
                            continue
                        repair_candidates.append(
                            {
                                "block_id": block.id,
                                "x": row,
                                "y": col,
                                "height": block.x,
                                "width": block.y,
                                "score": self._anchor_score(block, row, col),
                                "congestion_score": self._cluster_congestion_score(block, row, col),
                                "priority_score": round(self._block_priority_score(block), 4),
                                "action_type": "move",
                            }
                        )
                self.canvas.occupy_region(original_position, block.y, block.x, block_num)
                repair_candidates.sort(key=lambda item: (-item["score"], item["x"], item["y"]))
                candidates.extend(repair_candidates[:per_block_limit])

            candidates.sort(key=lambda item: (item.get("action_type", "place"), -item["priority_score"], -item["score"], item["block_id"], item["x"], item["y"]))

        return candidates
    
    def _block_to_dict(self,block):
        return{
            "id":block.id,
            "height":block.x,
            "width":block.y,
            "type": block.type,
            "power": block.power,
            "fixed": block.fixed,
            "committed": bool(getattr(block, "committed", False)),
            "move_count": int(self._long_horizon_move_counts.get(block.id, 0)),
            "placed":block.placed,
            "position":block.position
        }
    
    def _cluster_congestion_score(self,block:"Block",x:int,y:int) -> float:
        candidate_center = (x + (block.x / 2.0), y + (block.y / 2.0))

        attraction = 0.0
        total_weight = 0.0

        for neighbor_id, edge_info in block.get_internal_netlist().items():
            neighbor_block = self._block_lookup.get(neighbor_id)
            if neighbor_block is None or not neighbor_block.placed:
                continue
            neighbor_center = self._block_center(neighbor_block)
            if neighbor_center is None:
                continue

            importance = self._display_edge_importance(edge_info)
            dist = max(1.0, self._manhattan_distance(candidate_center, neighbor_center))
            attraction += importance / dist
            total_weight += importance

        if total_weight > 0:
            attraction /= total_weight
        else:
            grid_center = (self.grid_size / 2.0, self.grid_size / 2.0)
            attraction = 1.0 - (self._manhattan_distance(candidate_center, grid_center) / max(1.0, float(self.grid_size)))

        density = self._coarse_density_map()
        r = min(len(density) - 1, max(0, int(x * len(density) / self.grid_size)))
        c = min(len(density[0]) - 1, max(0, int(y * len(density[0]) / self.grid_size)))
        congestion = density[r][c] * (1.0 + (0.15 * block.power))

        return round(attraction - (0.7 * congestion), 4)







    def _render_ascii_board(self) -> str:
        if self.canvas is None:
            return "Reset the environment to view the board."

        id_by_number = {number: block_id for block_id, number in self.block_id_map.items()}
        size = self.grid_size
        cell_width = 2
        lines = ["y"]

        for row in range(size - 1, -1, -1):
            row_cells = []
            for col in range(size):
                cell = self.canvas.grid[row][col]
                if cell == 0:
                    row_cells.append(" " * cell_width)
                else:
                    row_cells.append(f"{id_by_number.get(cell, cell):>{cell_width}}")
            lines.append(f"{row:>2} |" + "".join(row_cells).rstrip())

        axis = "   +" + "-" * (size * cell_width)
        labels = "    " + "".join(f"{col:>{cell_width}}" for col in range(size)) + " (x)"
        lines.append(axis)
        lines.append(labels)
        return "\n".join(lines)
    
    def _build_observation(self,invalid_reason: Optional[str]=None)->ChipFlooringObservation:
        remaining_block_summaries = [self._block_summary(b) for b in self._state.remaining_blocks]
        if remaining_block_summaries:
            focus_block = remaining_block_summaries[0]
        elif self._is_long_horizon_task() and self.phase in {"repair", "finalize"}:
            repair_blocks = [
                b for b in self._state.placed_blocks
                if not b.fixed
            ]
            repair_blocks.sort(
                key=lambda b: (
                    -self._block_priority_score(b),
                    -(b.power),
                    b.id,
                )
            )
            remaining_block_summaries = [self._block_summary(b) for b in repair_blocks[:6]]
            focus_block = remaining_block_summaries[0] if remaining_block_summaries else None
        else:
            focus_block = None
        return ChipFlooringObservation(
            canva_space=self.canvas.grid,
            board_ascii=self._render_ascii_board(),
            remaining_blocks=[self._block_to_dict(b) for b in self._state.remaining_blocks],
            placed_blocks=[self._block_to_dict(b) for b in self._state.placed_blocks],
            block_summaries=remaining_block_summaries,
            candidate_positions=self._generate_candidate_positions(),
            density_map=self._coarse_density_map(),
            placement_focus=focus_block,
            current_hpwl=self._state.current_hpwl,
            delta_hpwl=self._state.delta_hpwl,
            placed_block_count=len(self._state.placed_blocks),
            task_name=self._state.task_name,
            phase=self._state.phase,
            phase_step=self._state.phase_step,
            instruction=self._state.instruction,
            hidden_constraint_count=self._state.hidden_constraint_count,
            revealed_constraint_count=self._state.revealed_constraint_count,
            done=self._state.done,
            reward=self._state.reward,
            invalid_reasons=invalid_reason,
        )

    def _build_task_configs(self):
        configs = {
            "easy": {
                "grid_size": 12,
                "nodes": [
                    {"id": "A", "height": 2, "width": 1},
                    {"id": "B", "height": 2, "width": 2},
                    {"id": "C", "height": 1, "width": 3},
                    {"id": "D", "height": 2, "width": 1},
                    {"id": "E", "height": 1, "width": 2},
                ],
                "edges": [
                    {"from": "A", "to": "B", "weight": 1.4, "criticality": 0.9},
                    {"from": "A", "to": "C", "weight": 1.1, "criticality": 0.6},
                    {"from": "B", "to": "D", "weight": 1.6, "criticality": 0.8},
                    {"from": "C", "to": "E", "weight": 1.3, "criticality": 0.7},
                    {"from": "B", "to": "E", "weight": 0.9, "criticality": 0.5},
                ],
            },
            "medium": {
                "grid_size": 18,
                "nodes": [
                    {"id": "A", "height": 2, "width": 1},
                    {"id": "B", "height": 3, "width": 1},
                    {"id": "C", "height": 1, "width": 4},
                    {"id": "D", "height": 2, "width": 2},
                    {"id": "E", "height": 1, "width": 3},
                    {"id": "F", "height": 3, "width": 2},
                    {"id": "G", "height": 2, "width": 3},
                    {"id": "H", "height": 1, "width": 2},
                    {"id": "I", "height": 4, "width": 1},
                    {"id": "J", "height": 2, "width": 4},
                ],
                "edges": [
                    {"from": "A", "to": "F", "weight": 2.4, "criticality": 0.95},
                    {"from": "A", "to": "C", "weight": 1.1, "criticality": 0.55},
                    {"from": "B", "to": "G", "weight": 1.8, "criticality": 0.85},
                    {"from": "B", "to": "D", "weight": 0.9, "criticality": 0.4},
                    {"from": "C", "to": "H", "weight": 2.1, "criticality": 0.75},
                    {"from": "C", "to": "J", "weight": 1.4, "criticality": 0.65},
                    {"from": "D", "to": "I", "weight": 1.7, "criticality": 0.7},
                    {"from": "E", "to": "J", "weight": 2.6, "criticality": 0.9},
                    {"from": "B", "to": "E", "weight": 1.75, "criticality": 0.6},
                    {"from": "D", "to": "G", "weight": 2.05, "criticality": 0.8},
                    {"from": "F", "to": "J", "weight": 1.55, "criticality": 0.7},
                ],
            },
            "heterogeneous": {
                "grid_size": 24,
                "nodes": [
                    {"id": "A", "height": 2, "width": 1, "type": "macro", "power": 3.0},
                    {"id": "B", "height": 3, "width": 1, "type": "standard", "power": 1.0},
                    {"id": "C", "height": 1, "width": 4, "type": "standard", "power": 1.1},
                    {"id": "D", "height": 2, "width": 2, "type": "macro", "power": 2.5},
                    {"id": "E", "height": 1, "width": 3, "type": "standard", "power": 1.0},
                    {"id": "F", "height": 3, "width": 2, "type": "standard", "power": 1.2},
                    {"id": "G", "height": 2, "width": 3, "type": "macro", "power": 2.8},
                    {"id": "H", "height": 1, "width": 2, "type": "standard", "power": 0.9},
                    {"id": "I", "height": 4, "width": 1, "type": "standard", "power": 1.0},
                    {"id": "J", "height": 2, "width": 4, "type": "macro", "power": 3.2},
                    {"id": "K", "height": 3, "width": 2, "type": "standard", "power": 1.1},
                    {"id": "L", "height": 1, "width": 1, "type": "standard", "power": 0.8},
                ],
                "edges": [
                    {"from": "A", "to": "D", "weight": 2.4, "criticality": 0.95},
                    {"from": "A", "to": "G", "weight": 1.6, "criticality": 0.8},
                    {"from": "B", "to": "E", "weight": 1.2, "criticality": 0.6},
                    {"from": "B", "to": "H", "weight": 0.9, "criticality": 0.35},
                    {"from": "C", "to": "F", "weight": 1.8, "criticality": 0.7},
                    {"from": "C", "to": "J", "weight": 2.3, "criticality": 0.9},
                    {"from": "D", "to": "I", "weight": 1.5, "criticality": 0.65},
                    {"from": "D", "to": "K", "weight": 1.1, "criticality": 0.5},
                    {"from": "E", "to": "J", "weight": 2.0, "criticality": 0.85},
                    {"from": "F", "to": "L", "weight": 1.4, "criticality": 0.55},
                    {"from": "G", "to": "J", "weight": 2.6, "criticality": 0.9},
                    {"from": "H", "to": "K", "weight": 1.0, "criticality": 0.4},
                    {"from": "I", "to": "L", "weight": 1.3, "criticality": 0.55},
                    {"from": "K", "to": "L", "weight": 0.8, "criticality": 0.3},
                ],
            },
            "fixed_obstacles": {
                "grid_size": 24,
                "nodes": [
                    {"id": "A", "height": 2, "width": 1},
                    {"id": "B", "height": 3, "width": 1},
                    {"id": "C", "height": 1, "width": 4},
                    {"id": "D", "height": 2, "width": 2},
                    {"id": "E", "height": 1, "width": 3},
                    {"id": "F", "height": 3, "width": 2},
                    {"id": "G", "height": 2, "width": 3},
                    {"id": "H", "height": 1, "width": 2},
                    {"id": "I", "height": 4, "width": 1},
                    {"id": "J", "height": 2, "width": 4},
                    {"id": "K", "height": 3, "width": 2},
                    {"id": "L", "height": 1, "width": 1},
                    {"id": "M", "height": 2, "width": 1},
                    {"id": "N", "height": 1, "width": 2},
                    {"id": "O", "height": 3, "width": 3},
                    {"id": "P", "height": 2, "width": 2, "fixed": True, "position": [8, 8]},
                    {"id": "Q", "height": 1, "width": 4, "fixed": True, "position": [14, 4]},
                    {"id": "R", "height": 3, "width": 1, "fixed": True, "position": [4, 16]},
                ],
                "edges": [
                    {"from": "A", "to": "F", "weight": 2.4, "criticality": 0.95},
                    {"from": "A", "to": "C", "weight": 1.1, "criticality": 0.55},
                    {"from": "B", "to": "G", "weight": 1.8, "criticality": 0.85},
                    {"from": "B", "to": "D", "weight": 0.9, "criticality": 0.45},
                    {"from": "C", "to": "H", "weight": 2.1, "criticality": 0.75},
                    {"from": "C", "to": "J", "weight": 1.4, "criticality": 0.65},
                    {"from": "D", "to": "I", "weight": 1.7, "criticality": 0.7},
                    {"from": "D", "to": "K", "weight": 0.8, "criticality": 0.5},
                    {"from": "E", "to": "J", "weight": 2.6, "criticality": 0.9},
                    {"from": "E", "to": "L", "weight": 1.0, "criticality": 0.55},
                    {"from": "F", "to": "M", "weight": 1.2, "criticality": 0.6},
                    {"from": "F", "to": "N", "weight": 2.0, "criticality": 0.8},
                    {"from": "G", "to": "O", "weight": 2.8, "criticality": 0.95},
                    {"from": "B", "to": "E", "weight": 1.75, "criticality": 0.6},
                    {"from": "D", "to": "G", "weight": 2.05, "criticality": 0.8},
                    {"from": "F", "to": "J", "weight": 1.55, "criticality": 0.7},
                    {"from": "H", "to": "L", "weight": 0.85, "criticality": 0.45},
                    {"from": "I", "to": "N", "weight": 1.45, "criticality": 0.65},
                    {"from": "K", "to": "O", "weight": 2.25, "criticality": 0.9},
                    {"from": "P", "to": "A", "weight": 1.8, "criticality": 0.75},
                    {"from": "P", "to": "J", "weight": 2.1, "criticality": 0.9},
                    {"from": "Q", "to": "D", "weight": 1.3, "criticality": 0.55},
                    {"from": "Q", "to": "K", "weight": 1.6, "criticality": 0.7},
                    {"from": "R", "to": "G", "weight": 1.9, "criticality": 0.8},
                    {"from": "R", "to": "O", "weight": 2.2, "criticality": 0.85},
                ],
            },
            "long_horizon": {
                "grid_size": 28,
                "phase_reveal_step": 10,
                "phase_repair_step": 20,
                "phase_finalize_step": 30,
                "hidden_edge_fraction": 0.35,
                "nodes": [
                    {"id": "A", "height": 2, "width": 1, "type": "macro", "power": 3.2},
                    {"id": "B", "height": 3, "width": 1, "type": "standard", "power": 1.1},
                    {"id": "C", "height": 1, "width": 4, "type": "standard", "power": 1.0},
                    {"id": "D", "height": 2, "width": 2, "type": "macro", "power": 2.7},
                    {"id": "E", "height": 1, "width": 3, "type": "standard", "power": 1.0},
                    {"id": "F", "height": 3, "width": 2, "type": "standard", "power": 1.2},
                    {"id": "G", "height": 2, "width": 3, "type": "macro", "power": 2.9},
                    {"id": "H", "height": 1, "width": 2, "type": "standard", "power": 0.9},
                    {"id": "I", "height": 4, "width": 1, "type": "standard", "power": 1.0},
                    {"id": "J", "height": 2, "width": 4, "type": "macro", "power": 3.4},
                    {"id": "K", "height": 3, "width": 2, "type": "standard", "power": 1.1},
                    {"id": "L", "height": 1, "width": 1, "type": "standard", "power": 0.8},
                    {"id": "M", "height": 2, "width": 1, "type": "standard", "power": 1.0},
                    {"id": "N", "height": 1, "width": 2, "type": "standard", "power": 0.9},
                    {"id": "O", "height": 3, "width": 3, "type": "macro", "power": 3.0},
                    {"id": "P", "height": 2, "width": 2, "type": "standard", "power": 1.2},
                    {"id": "Q", "height": 1, "width": 3, "type": "standard", "power": 1.0},
                    {"id": "R", "height": 3, "width": 1, "type": "standard", "power": 1.1},
                    {"id": "S", "height": 2, "width": 2, "type": "macro", "power": 2.6},
                    {"id": "T", "height": 1, "width": 4, "type": "standard", "power": 1.0},
                    {"id": "U", "height": 2, "width": 1, "type": "standard", "power": 1.0},
                    {"id": "V", "height": 3, "width": 2, "type": "standard", "power": 1.1},
                    {"id": "W", "height": 1, "width": 2, "type": "standard", "power": 0.9},
                    {"id": "X", "height": 2, "width": 3, "type": "macro", "power": 3.1},
                ],
                "edges": [
                    {"from": "A", "to": "D", "weight": 2.6, "criticality": 0.95},
                    {"from": "A", "to": "G", "weight": 1.9, "criticality": 0.85},
                    {"from": "A", "to": "J", "weight": 2.4, "criticality": 0.9},
                    {"from": "B", "to": "E", "weight": 1.3, "criticality": 0.6},
                    {"from": "B", "to": "H", "weight": 1.0, "criticality": 0.4},
                    {"from": "C", "to": "F", "weight": 1.7, "criticality": 0.7},
                    {"from": "C", "to": "I", "weight": 1.1, "criticality": 0.5},
                    {"from": "D", "to": "J", "weight": 2.3, "criticality": 0.9},
                    {"from": "D", "to": "K", "weight": 1.0, "criticality": 0.45},
                    {"from": "E", "to": "L", "weight": 0.9, "criticality": 0.35},
                    {"from": "E", "to": "M", "weight": 1.2, "criticality": 0.55},
                    {"from": "F", "to": "N", "weight": 2.0, "criticality": 0.8},
                    {"from": "G", "to": "O", "weight": 2.8, "criticality": 0.95},
                    {"from": "H", "to": "P", "weight": 1.5, "criticality": 0.65},
                    {"from": "I", "to": "Q", "weight": 1.2, "criticality": 0.55},
                    {"from": "J", "to": "R", "weight": 1.9, "criticality": 0.75},
                    {"from": "K", "to": "S", "weight": 2.1, "criticality": 0.85},
                    {"from": "L", "to": "T", "weight": 0.8, "criticality": 0.3},
                    {"from": "M", "to": "U", "weight": 1.4, "criticality": 0.6},
                    {"from": "N", "to": "V", "weight": 1.6, "criticality": 0.7},
                    {"from": "O", "to": "W", "weight": 2.2, "criticality": 0.88},
                    {"from": "P", "to": "X", "weight": 2.0, "criticality": 0.82},
                    {"from": "Q", "to": "X", "weight": 1.5, "criticality": 0.65},
                    {"from": "S", "to": "X", "weight": 2.4, "criticality": 0.93},
                    {"from": "R", "to": "T", "weight": 1.0, "criticality": 0.4},
                    {"from": "V", "to": "X", "weight": 1.7, "criticality": 0.78},
                ],
            },
            "hard": {
                "grid_size": 24,
                "nodes": [
                    {"id": "A", "height": 2, "width": 1},
                    {"id": "B", "height": 3, "width": 1},
                    {"id": "C", "height": 1, "width": 4},
                    {"id": "D", "height": 2, "width": 2},
                    {"id": "E", "height": 1, "width": 3},
                    {"id": "F", "height": 3, "width": 2},
                    {"id": "G", "height": 2, "width": 3},
                    {"id": "H", "height": 1, "width": 2},
                    {"id": "I", "height": 4, "width": 1},
                    {"id": "J", "height": 2, "width": 4},
                    {"id": "K", "height": 3, "width": 2},
                    {"id": "L", "height": 1, "width": 1},
                    {"id": "M", "height": 2, "width": 1},
                    {"id": "N", "height": 1, "width": 2},
                    {"id": "O", "height": 3, "width": 3},
                ],
                "edges": [
                    {"from": "A", "to": "F", "weight": 2.4, "criticality": 0.95},
                    {"from": "A", "to": "C", "weight": 1.1, "criticality": 0.55},
                    {"from": "B", "to": "G", "weight": 1.8, "criticality": 0.85},
                    {"from": "B", "to": "D", "weight": 0.9, "criticality": 0.45},
                    {"from": "C", "to": "H", "weight": 2.1, "criticality": 0.75},
                    {"from": "C", "to": "J", "weight": 1.4, "criticality": 0.65},
                    {"from": "D", "to": "I", "weight": 1.7, "criticality": 0.7},
                    {"from": "D", "to": "K", "weight": 0.8, "criticality": 0.5},
                    {"from": "E", "to": "J", "weight": 2.6, "criticality": 0.9},
                    {"from": "E", "to": "L", "weight": 1.0, "criticality": 0.55},
                    {"from": "F", "to": "M", "weight": 1.2, "criticality": 0.6},
                    {"from": "F", "to": "N", "weight": 2.0, "criticality": 0.8},
                    {"from": "G", "to": "O", "weight": 2.8, "criticality": 0.95},
                    {"from": "B", "to": "E", "weight": 1.75, "criticality": 0.6},
                    {"from": "D", "to": "G", "weight": 2.05, "criticality": 0.8},
                    {"from": "F", "to": "J", "weight": 1.55, "criticality": 0.7},
                    {"from": "H", "to": "L", "weight": 0.85, "criticality": 0.45},
                    {"from": "I", "to": "N", "weight": 1.45, "criticality": 0.65},
                    {"from": "K", "to": "O", "weight": 2.25, "criticality": 0.9},
                ],
            },
        }
        composite_configs = self._build_composite_task_configs(configs)
        composite_configs["long_horizon"] = self._clone_task_config(composite_configs["hard_heterogeneous_long_horizon"])
        return composite_configs

    def _select_task_netlist(self, task_name: str):
        resolved = self._resolve_task_name(task_name)
        config = self.task_configs.get(resolved, self.task_configs["hard_standard_long_horizon"])
        return {
            "nodes": list(config["nodes"]),
            "edges": list(config["edges"]),
        }

    def _select_task_grid_size(self, task_name: str) -> int:
        resolved = self._resolve_task_name(task_name)
        config = self.task_configs.get(resolved, self.task_configs["hard_standard_long_horizon"])
        return int(config["grid_size"])
    


class Canvas:

    '''
        Initializing the grid type canvas in order to pricisely map the components in the canvas
    '''
    def __init__(self,grid_size):
        self.grid_size = grid_size
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    '''
        Function for knowing whether the particular unit is available or not
    '''

    def is_unit_occupied(self, row, col):
        return self.grid[row][col] != 0
    
    '''
        Function to identify whether the component can be placed in the grid
    '''

    def can_occupy(self, position, width, height):
        row, col = position

        # boundary check
        if row + height > self.grid_size or col + width > self.grid_size:
            return False

        # overlap check
        for dx in range(height):
            for dy in range(width):
                if self.grid[row + dx][col + dy] != 0:
                    return False

        return True
        
    '''
        Function for using the group of cords in the grid
    '''
    def occupy_region(self, position, width, height, block_id):
        row, col = position
        for dx in range(height):
            for dy in range(width):
                self.grid[row + dx][col + dy] = block_id

    '''
        Function for removing the group of cords in the grid
    '''
    
    def remove_region(self, position, width, height):
        row, col = position
        for dx in range(height):
            for dy in range(width):
                self.grid[row + dx][col + dy] = 0

class Block:
    def __init__(self,id,height,width,block_type="standard",power=1.0,fixed=False):
        self.id = id 
        self.x  = height
        self.y  = width
        self.type = block_type
        self.power = power
        self.fixed = fixed
        self.committed = False
        self.move_count = 0
        self.placed = False
        self.position = None
        self.internal_netlist = {}

    def connect_block(self,block,weight,criticality=0.0,edge_key=None):
        self.internal_netlist[block.id] = {
            "weight": float(weight),
            "criticality": float(criticality),
            "edge_key": tuple(edge_key) if edge_key is not None else None,
        }
    
    def get_internal_netlist(self):
        return self.internal_netlist

        


       

            
           

    
 

        
