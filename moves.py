from dataclasses import dataclass
from consts import Coord, Wall


@dataclass(frozen=True)
class Move:
    player: int


@dataclass(frozen=True)
class PlayerMove(Move):
    coord: Coord


@dataclass(frozen=True)
class WallMove(Move):
    wall: Wall
