from uuid import uuid4
import os
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
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_name = os.getenv("TASK_NAME", "hard").strip().lower() or "hard"
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
        self.task_configs = self._build_task_configs()
        self.global_netlist = self._select_task_netlist(self.task_name)
        self.grid_size = self._select_task_grid_size(self.task_name)
 


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
        self._state = ChipFlooringResponseState(
            episode_id=str(uuid4()),
            step_count=0,
            grid_size=self.grid_size,
            grid=self.canvas.grid,
            blocks=self.blocks,
            placed_blocks=[],
            remaining_blocks=list(self.blocks),
            done=False,
            reward=0,
            current_hpwl=0.0,
            delta_hpwl=0.0,
            task_name=self.task_name,
        )

        return self._build_observation()

         
    def step(self, action: ChipFlooringAction) -> ChipFlooringObservation:  # type: ignore[override]
 
        self._state.step_count += 1
        self._state.reward=0.0
        self._state.done=False
        self._state.delta_hpwl = 0.0
        
        invalid_reasons = None
        x = action.x
        y = action.y
        current_block_index = action.choosen_block_index

        if not isinstance(current_block_index,int) or current_block_index < 0 or current_block_index >= len(self._state.blocks):
            invalid_reasons = "Invalid Block Index correclty choose the block index with in the range"
            self._state.reward=self.invalid_block_penalty
        else:
            block = self._state.blocks[current_block_index]
            
            if block not in self._state.remaining_blocks:
                invalid_reasons="The selected block is not in the ramining block properly choose the correct block"
                self._state.reward=self.invalid_block_penalty
            elif not self.canvas.can_occupy((x,y),block.y,block.x):
                invalid_reasons = "The given possition cannot be occupied check the canvas once again for the right placment"
                self._state.reward = self.invalid_bounds_penalty

            else:
                block_num = self.block_id_map[block.id]
                self.canvas.occupy_region((x,y),block.y,block.x,block_num)
                block.placed=True
                block.position=(x,y)
                self._state.placed_blocks.append(block)
                self._state.remaining_blocks.remove(block)
                incremental_hpwl = self._compute_incremental_hpwl(block)
                placed_neighbor_weight = self._placed_neighbor_weight(block)
                self._state.delta_hpwl = incremental_hpwl
                self._state.current_hpwl = self._compute_total_hpwl()
                self._state.reward = (
                    self.valid_placement_bonus
                    + (0.02 * placed_neighbor_weight)
                    - (self.hpwl_weight * incremental_hpwl)
                )
                self._state.done = len(self._state.remaining_blocks) == 0
                if self._state.done:
                    self._state.reward += self.final_completion_bonus - (self.hpwl_weight * self._state.current_hpwl)
        
        self._state.grid = self.canvas.grid
        self._state.trajectory.append(
            {
                "step_count": self._state.step_count,
                "action": {
                    "x": x,
                    "y": y,
                    "choosen_block_index": current_block_index,
                },
                "reward": self._state.reward,
                "done": self._state.done,
                "invalid_reason": invalid_reasons,
                "current_hpwl": self._state.current_hpwl,
                "delta_hpwl": self._state.delta_hpwl,
                "task_name": self._state.task_name,
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
            block = Block(id=node["id"], height=node["height"], width=node["width"])
            blocks[node["id"]] = block
        
        for edge in edges:
            src = edge["from"]
            dist = edge["to"]
            weight = edge["weight"]
            blocks[src].connect_block(blocks[dist], weight)
            blocks[dist].connect_block(blocks[src], weight)

        return list(blocks.values())

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
        for neighbor_id, weight in placed_block.get_internal_netlist().items():
            neighbor = self._block_lookup.get(neighbor_id)
            if neighbor is None or not neighbor.placed:
                continue
            neighbor_center = self._block_center(neighbor)
            if neighbor_center is None:
                continue
            total += weight * self._manhattan_distance(placed_center, neighbor_center) / self.grid_size

        return total

    def _placed_neighbor_weight(self, placed_block: "Block") -> float:
        total = 0.0
        for neighbor_id, weight in placed_block.get_internal_netlist().items():
            neighbor = self._block_lookup.get(neighbor_id)
            if neighbor is not None and neighbor.placed:
                total += weight
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
            total += edge["weight"] * self._manhattan_distance(src_center, dst_center) / self.grid_size

        return total

    def _block_priority_score(self, block: "Block") -> float:
        placed_neighbor_weight = 0.0
        placed_neighbor_count = 0
        for neighbor_id, weight in block.get_internal_netlist().items():
            neighbor = self._block_lookup.get(neighbor_id)
            if neighbor is not None and neighbor.placed:
                placed_neighbor_weight += weight
                placed_neighbor_count += 1
        degree = len(block.get_internal_netlist())
        area = block.x * block.y
        return (placed_neighbor_weight * 10.0) + (placed_neighbor_count * 2.0) + (degree * 0.25) + (area * 0.05)

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

        for neighbor_id, weight in sorted(
            block.get_internal_netlist().items(),
            key=lambda item: (-item[1], item[0]),
        ):
            neighbor = self._block_lookup.get(neighbor_id)
            neighbor_summary = {
                "id": neighbor_id,
                "weight": weight,
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
            "area": block.x * block.y,
            "degree": len(block.get_internal_netlist()),
            "priority_score": self._block_priority_score(block),
            "connected_blocks": [neighbor["id"] for neighbor in neighbors],
            "strongest_neighbors": strongest_neighbors,
            "placed_neighbors": placed_neighbors,
            "unplaced_neighbors": unplaced_neighbors,
            "placed_neighbor_count": len(placed_neighbors),
            "placed_neighbor_weight": sum(n["weight"] for n in placed_neighbors),
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
        for neighbor_id, weight in block.get_internal_netlist().items():
            neighbor = self._block_lookup.get(neighbor_id)
            if neighbor is None or not neighbor.placed:
                continue
            neighbor_center = self._block_center(neighbor)
            if neighbor_center is None:
                continue
            score -= weight * self._manhattan_distance(center, neighbor_center) / self.grid_size
            placed_neighbor_count += 1

        if placed_neighbor_count == 0:
            grid_center = (self.grid_size / 2.0, self.grid_size / 2.0)
            score -= 0.05 * self._manhattan_distance(center, grid_center) / self.grid_size

        density = self._coarse_density_map()
        density_row = min(len(density) - 1, max(0, row * len(density) // self.grid_size))
        density_col = min(len(density[0]) - 1, max(0, col * len(density[0]) // self.grid_size))
        score -= 0.15 * density[density_row][density_col]

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
                            "priority_score": round(self._block_priority_score(block), 4),
                        }
                    )
            block_candidates.sort(key=lambda item: (-item["score"], item["x"], item["y"]))
            candidates.extend(block_candidates[:per_block_limit])

        candidates.sort(key=lambda item: (-item["priority_score"], -item["score"], item["block_id"], item["x"], item["y"]))
        return candidates
    
    def _block_to_dict(self,block):
        return{
            "id":block.id,
            "height":block.x,
            "width":block.y,
            "placed":block.placed,
            "position":block.position
        }
    
    def _build_observation(self,invalid_reason: Optional[str]=None)->ChipFlooringObservation:
        remaining_block_summaries = [self._block_summary(b) for b in self._state.remaining_blocks]
        focus_block = remaining_block_summaries[0] if remaining_block_summaries else None
        return ChipFlooringObservation(
            canva_space=self.canvas.grid,
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
            done=self._state.done,
            reward=self._state.reward,
            invalid_reasons=invalid_reason,
        )

    def _build_task_configs(self):
        return {
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
                    {"from": "A", "to": "B", "weight": 1.4},
                    {"from": "A", "to": "C", "weight": 1.1},
                    {"from": "B", "to": "D", "weight": 1.6},
                    {"from": "C", "to": "E", "weight": 1.3},
                    {"from": "B", "to": "E", "weight": 0.9},
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
                    {"from": "A", "to": "F", "weight": 2.4},
                    {"from": "A", "to": "C", "weight": 1.1},
                    {"from": "B", "to": "G", "weight": 1.8},
                    {"from": "B", "to": "D", "weight": 0.9},
                    {"from": "C", "to": "H", "weight": 2.1},
                    {"from": "C", "to": "J", "weight": 1.4},
                    {"from": "D", "to": "I", "weight": 1.7},
                    {"from": "E", "to": "J", "weight": 2.6},
                    {"from": "B", "to": "E", "weight": 1.75},
                    {"from": "D", "to": "G", "weight": 2.05},
                    {"from": "F", "to": "J", "weight": 1.55},
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
                    {"from": "A", "to": "F", "weight": 2.4},
                    {"from": "A", "to": "C", "weight": 1.1},
                    {"from": "B", "to": "G", "weight": 1.8},
                    {"from": "B", "to": "D", "weight": 0.9},
                    {"from": "C", "to": "H", "weight": 2.1},
                    {"from": "C", "to": "J", "weight": 1.4},
                    {"from": "D", "to": "I", "weight": 1.7},
                    {"from": "D", "to": "K", "weight": 0.8},
                    {"from": "E", "to": "J", "weight": 2.6},
                    {"from": "E", "to": "L", "weight": 1.0},
                    {"from": "F", "to": "M", "weight": 1.2},
                    {"from": "F", "to": "N", "weight": 2.0},
                    {"from": "G", "to": "O", "weight": 2.8},
                    {"from": "B", "to": "E", "weight": 1.75},
                    {"from": "D", "to": "G", "weight": 2.05},
                    {"from": "F", "to": "J", "weight": 1.55},
                    {"from": "H", "to": "L", "weight": 0.85},
                    {"from": "I", "to": "N", "weight": 1.45},
                    {"from": "K", "to": "O", "weight": 2.25},
                ],
            },
        }

    def _select_task_netlist(self, task_name: str):
        config = self.task_configs.get(task_name, self.task_configs["hard"])
        return {
            "nodes": list(config["nodes"]),
            "edges": list(config["edges"]),
        }

    def _select_task_grid_size(self, task_name: str) -> int:
        config = self.task_configs.get(task_name, self.task_configs["hard"])
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

    def is_unit_occupied(self,x,y):
        return self.grid[x][y] != 0
    
    '''
        Function to identify whether the component can be placed in the grid
    '''

    def can_occupy(self, anchor, width, height):
        row, col = anchor

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
    def occupy_region(self, anchor, width, height, block_id):
        row, col = anchor
        for dx in range(height):
            for dy in range(width):
                self.grid[row + dx][col + dy] = block_id

    '''
        Function for removing the group of cords in the grid
    '''
    
    def remove_region(self, anchor, width, height):
        row, col = anchor
        for dx in range(height):
            for dy in range(width):
                self.grid[row + dx][col + dy] = 0

class Block:
    def __init__(self,id,height,width):
        self.id = id 
        self.x  = height
        self.y  = width
        self.placed = False
        self.position = None
        self.internal_netlist = {}

    def connect_block(self,block,weight):
        self.internal_netlist[block.id] = weight
    
    def get_internal_netlist(self):
        return self.internal_netlist

        


       

            
           

    
 

        
