from __future__ import annotations
from collections import deque
from functools import reduce
from typing import Callable, Iterator, Optional

import numpy as np

from .functions import reduce_sum, restrict_mean, refine_donor
from .typing import IntegerArray, FloatArray
from .typing import ReduceFunction, RestrictFunction, RefineFunction

__all__ = [
    'ReductionTree',
    'ReductionNode',
    'ReductionNodeList'
]


class ReductionNodeList:
    list: list[Optional[ReductionNode]]
    length: int
    iter: int

    def __init__(self) -> None:
        self.list = []
        self.length = 0

    def __repr__(self) -> str:
        return f'{self.list}'

    def __bool__(self) -> bool:
        return self.length > 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[ReductionNode]:
        self.iter = 0
        return self

    def __next__(self) -> ReductionNode:
        while self.iter < len(self.list):
            node = self.list[self.iter]
            self.iter += 1
            if node is not None:
                return node
        raise StopIteration

    def __setitem__(self, key: IntegerArray,
                    value: Optional[ReductionNode]) -> None:
        while len(self.list) < (1 << len(key)):
            self.list.append(None)
        idx = self.flatten(key)
        self.length += (value is not None) != (self.list[idx] is not None)
        self.list[idx] = value

    def __getitem__(self, key: IntegerArray) -> Optional[ReductionNode]:
        while len(self.list) < (1 << len(key)):
            self.list.append(None)
        return self.list[self.flatten(key)]

    @staticmethod
    def flatten(key: IntegerArray) -> int:
        return reduce(lambda a, b: a * 2 + b, key, 0)


class ReductionNode:
    tree: ReductionTree
    level: int
    lloc: IntegerArray
    value: FloatArray
    value_leaf: FloatArray
    parent: Optional[ReductionNode]
    child: ReductionNodeList

    def __init__(self, tree: ReductionTree, level: int,
                 lloc: IntegerArray,) -> None:
        self.tree = tree
        self.level = level
        self.lloc = lloc

        self.value = np.full(tree.shape, tree.start)
        self.value_leaf = np.full(tree.shape, tree.start)

        self.parent = None
        self.child = ReductionNodeList()

    def __repr__(self) -> str:
        return (f'ReductionNode(level={self.level}, lloc={self.lloc.tolist()},'
                f' child={self.child})')

    def add(self, value: FloatArray, *, slc: tuple[slice, ...] = None,
            skip_value: bool = False) -> None:
        if slc is None:
            slc = tuple(slice(0, s) for s in self.tree.shape)
            if not skip_value:
                self.value[...] = self.tree.reduce_func(self.value, value)
        else:
            self.value_leaf[slc] = self.tree.reduce_func(
                self.value_leaf[slc], value)

        if self.parent is not None:
            slc_unrestricted = tuple(
                slice(s + sc.start, s + sc.stop) if ll & 1 else sc
                for ll, s, sc in zip(self.lloc, self.tree.shape, slc))
            slc_restricted = tuple(
                slice(sc.start // 2, (sc.stop + 1) // 2)
                for sc in slc_unrestricted)
            padding = tuple(
                (sc_u.start - sc_r.start * 2, sc_r.stop * 2 - sc_u.stop)
                for sc_u, sc_r in zip(slc_unrestricted, slc_restricted))
            if any(any(p) for p in padding):
                value_padded = np.pad(
                    value, padding, constant_values=self.tree.start)
            else:
                value_padded = value
            value_restricted = self.tree.restrict_func(value_padded)
            self.parent.add(value_restricted, slc=slc_restricted)

    def get(self, *, slc: tuple[slice, ...] = None) -> FloatArray:
        if slc is None:
            slc = tuple(slice(0, s) for s in self.tree.shape)
            value = self.tree.reduce_func(self.value, self.value_leaf)
        else:
            value = self.value[slc]

        if self.parent is not None:
            slc_unrestricted = tuple(
                slice(s + sc.start, s + sc.stop) if ll & 1 else sc
                for ll, s, sc in zip(self.lloc, self.tree.shape, slc))
            slc_restricted = tuple(
                slice(sc.start // 2, (sc.stop + 1) // 2)
                for sc in slc_unrestricted)

            value_unrefined = self.parent.get(slc=slc_restricted)
            value_refined = self.tree.refine_func(value_unrefined)

            trimming = tuple(
                (sc_u.start - sc_r.start * 2, sc_r.stop * 2 - sc_u.stop)
                for sc_u, sc_r in zip(slc_unrestricted, slc_restricted))
            if any(any(t) for t in trimming):
                slc_trim = tuple(
                    slice(trim[0], s-trim[1])
                    for trim, s in zip(trimming, value_refined.shape)
                )
                value_trimmed = value_refined[slc_trim]
            else:
                value_trimmed = value_refined
            return self.tree.reduce_func(value, value_trimmed)
        return value


class ReductionTree:
    root: Optional[ReductionNode]
    shape: tuple[int, ...]
    start: float

    # TODO: check for mypy support
    # mypy do not regonize field of Callable type
    # see https://github.com/python/mypy/issues/708
    reduce_func: ReduceFunction | Callable
    restrict_func: RestrictFunction | Callable
    refine_func: RefineFunction | Callable

    def __init__(self, shape: tuple[int, ...], *, start: float = 0.0,
                 reduce_func: ReduceFunction = reduce_sum,
                 restrict_func: RestrictFunction = restrict_mean,
                 refine_func: RefineFunction = refine_donor) -> None:
        self.root = None
        self.shape = shape
        self.start = start

        self.reduce_func = reduce_func
        self.restrict_func = restrict_func
        self.refine_func = refine_func

    def __repr__(self) -> str:
        return f'ReductionTree(shape={self.shape}, root={self.root!r})'

    def insert(self, level: int, lloc: IntegerArray) -> ReductionNode:
        if self.root is None:
            self.root = ReductionNode(self, level, lloc)
            return self.root

        # go up until inserted node is in current subtree
        while (self.root.level > level or
                any(self.root.lloc != (lloc >> (level - self.root.level)))):
            old_root = self.root
            new_root = ReductionNode(
                self, old_root.level-1, old_root.lloc >> 1)
            new_root.child[old_root.lloc & 1] = old_root
            old_root.parent = new_root
            self.root = new_root
            value = self.reduce_func(old_root.value, old_root.value_leaf)
            old_root.add(value, skip_value=True)

        # traverse according to lloc
        node = self.root
        while node.level < level:
            new_lloc = (lloc >> (level - node.level - 1))
            key = new_lloc & 1
            new_node = node.child[key]
            if new_node is None:
                new_node = node.child[key] = ReductionNode(
                    self, node.level+1, new_lloc)
                new_node.parent = node
            node = new_node
        return node

    def all_nodes(self, leaf: bool = False) -> Iterator[ReductionNode]:
        stack: deque[ReductionNode] = deque()
        if self.root is not None:
            stack.append(self.root)
        while stack:
            node = stack.pop()
            if not (leaf and node.child):
                yield node
            for n in node.child:
                stack.append(n)
