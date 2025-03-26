from __future__ import annotations
import json
import pickle
import numpy as np

from .. import *
from tools.draw import EnvironmentDrawer


class Environment:
    def __init__(self, name: str, grid_dims: np.ndarray[int], grid_allowed: np.ndarray[bool] = None,
                 providers: tuple[Station] = None, receivers: tuple[Station] = None,
                 grid_size: np.ndarray[int] = None, mover_size: np.ndarray[int] = None, field_size: np.ndarray[int] = None):

        self.name = name

        self.grid_dims = grid_dims
        self.grid_allowed = grid_allowed.reshape(grid_dims, order='F') if grid_allowed is not None else np.full(self.grid_dims, True)

        self.providers = providers or tuple()
        self.receivers = receivers or tuple()

        self.grid_size = grid_size if grid_size is not None else np.array([120, 120], dtype=float)
        self.mover_size = mover_size if mover_size is not None else np.array([118, 118], dtype=float)
        self.field_size = field_size if field_size is not None else self.grid_dims * self.grid_size

        self.items: set[Item] = set([Item.empty()])
        for station in self.stations:
            self.items.update(station.items)

        self.drawer = EnvironmentDrawer(env=self)

    @property
    def stations(self) -> tuple[Station]:
        return self.providers + self.receivers

    def get_item(self, i: int | str) -> Item:
        return next(item for item in self.items if item == (int)(i))

    @classmethod
    def from_json(cls, filename: str) -> Environment:
        with open(filename) as f:
            data = json.load(f)

        name = filename

        grid_dims = data.get("grid", {}).get("dims", None)
        if grid_dims is not None:
            grid_dims = np.array(grid_dims, dtype=int)
        grid_size = data.get("grid", {}).get("size", None)
        if grid_size is not None:
            grid_size = np.array(grid_size, dtype=int)
        grid_allowed = np.array(data.get("grid", {}).get("allowed", np.full(grid_dims, True)), dtype=bool)

        field_size = data.get("field", {}).get("size", None)
        if field_size is not None:
            field_size = np.array(field_size, dtype=float)

        match data.get("pos_units", "mm"):
            case "grid":
                def parse_pos(pos: list[float]) -> np.ndarray[float]: return (np.array(pos, dtype=float) + (0.5, 0.5)) * grid_size
            case "mm" | _:
                def parse_pos(pos: list[float]) -> np.ndarray[float]: return np.array(pos, dtype=float)

        mover_size = np.array(data["movers"].get("size", [118, 118]), dtype=float)

        providers = tuple(Station(parse_pos(s["pos"]), tuple(Item(i) for i in s["items"]), s["item_type"], s["period"], s["duration"], s.get("reset", 0.0),  True,
                                  np.array(s["size"], dtype=float)) for s in data["providers"])

        receivers = tuple(Station(parse_pos(s["pos"]), tuple(Item(i) for i in s["items"]), s["item_type"], s["period"], s["duration"], s.get("reset", 0.0), False,
                                  np.array(s["size"], dtype=float)) for s in data["receivers"])

        return cls(name=name, grid_dims=grid_dims, grid_allowed=grid_allowed,
                   providers=providers, receivers=receivers,
                   grid_size=grid_size, mover_size=mover_size, field_size=field_size)

    def to_pickle(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filename: str) -> Environment:
        with open(filename, 'rb') as f:
            return pickle.load(f)
