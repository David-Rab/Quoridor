from dataclasses import dataclass
from consts import Coord, Wall


@dataclass(frozen=True)
class Move:
    player: str


@dataclass(frozen=True)
class PlayerMove(Move):
    coord: Coord


@dataclass(frozen=True)
class WallMove(Move):
    wall: Wall
