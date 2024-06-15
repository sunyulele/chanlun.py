"""
# -*- coding: utf-8 -*-
# @Time    : 2024/04/28 16:45
# @Author  : YuYuKunKun
# @File    : chan.py
"""

import json
import math
import struct
import asyncio
import time
import traceback

from pathlib import Path
from random import choice
from threading import Thread
from typing import (
    List,
    Union,
    Self,
    Literal,
    Optional,
    Tuple,
    final,
    Dict,
    Iterable,
    Any,
    Annotated,
)
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import reload
from enum import Enum
from abc import ABCMeta, abstractmethod, ABC

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    from termcolor import colored
except ImportError:

    def colored(text, color="red", on_color=None, attrs=None):
        """彩色字"""
        return text


ts2int = lambda timestamp_str: int(
    time.mktime(datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timetuple())
)


class Shape(Enum):
    """
    缠论分型
    """

    D = "底分型"
    G = "顶分型"
    S = "上升分型"
    X = "下降分型"
    T = "喇叭口型"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Direction(Enum):
    Up = "向上"
    Down = "向下"
    JumpUp = "缺口向上"
    JumpDown = "缺口向下"

    Left = "左包右"  # 顺序包含
    Right = "右包左"  # 逆序包含

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Freq(Enum):
    # 60 180 300 900 1800 3600 7200 14400 21600 43200 86400 259200
    S1: int = 1
    S3: int = 3
    S5: int = 5
    S12: int = 12
    m1: int = 60 * 1
    m3: int = 60 * 3
    m5: int = 60 * 5
    m15: int = 60 * 15
    m30: int = 60 * 30
    H1: int = 60 * 60 * 1
    H2: int = 60 * 60 * 2
    H4: int = 60 * 60 * 4
    D1: int = 60 * 60 * 24  # 86400
    D3: int = 60 * 60 * 24 * 3  # 259200


class ChanException(Exception):
    """exception"""

    ...


States = Literal["老阳", "少阴", "老阴", "少阳"]


def _print(*args, **kwords):
    result = []
    for i in args:
        if i in ("少阳", True, Shape.D, "底分型") or "少阳" in str(i):
            result.append(colored(i, "green"))

        elif i in ("老阳", False, Shape.G, "顶分型") or "老阳" in str(i):
            result.append(colored(i, "red"))

        elif i in ("少阴",) or "少阴" in str(i):
            result.append("\33[07m" + colored(i, "yellow"))

        elif i in ("老阴",) or "老阴" in str(i):
            result.append("\33[01m" + colored(i, "blue"))

        elif "PUSH" in str(i):
            result.append(colored(i, "red"))

        elif "POP" in str(i):
            result.append(colored(i, "green"))

        elif "ANALYSIS" in str(i):
            result.append(colored(i, "blue"))

        else:
            result.append(i)
    result = tuple(result)
    print(*result, **kwords)


def dp(*args, **kwords):
    if not 0:
        _print(*args, **kwords)


def bdp(*args, **kwargs):
    if not 1:
        dp(*args, **kwargs)


def ddp(*args, **kwargs):
    if not 1:
        dp(*args, **kwargs)


def zsdp(*args, **kwargs):
    if not 1:
        dp(*args, **kwargs)


class Pillar:
    def __init__(self, high: float, low: float):
        self.low = low
        self.high = high

    def __str__(self):
        return f"Pillar({self.high}, {self.low})"

    def __repr__(self):
        return f"Pillar({self.high}, {self.low})"


def double_relation(left, right) -> Direction:
    """
    两个带有[low, high]对象的所有关系
    """
    # assert hasattr(left, "low")
    # assert hasattr(left, "high")
    # assert hasattr(right, "low")
    # assert hasattr(right, "high")

    relation = None
    assert left is not right, ChanException("相同对象无法比较", left, right)

    if (left.low <= right.low) and (left.high >= right.high):
        relation = Direction.Left  # "左包右" # 顺序

    elif (left.low >= right.low) and (left.high <= right.high):
        relation = Direction.Right  # "右包左" # 逆序

    elif (left.low < right.low) and (left.high < right.high):
        relation = Direction.Up  # "上涨"
        if left.high < right.low:
            relation = Direction.JumpUp  # "跳涨"

    elif (left.low > right.low) and (left.high > right.high):
        relation = Direction.Down  # "下跌"
        if left.low > right.high:
            relation = Direction.JumpDown  # "跳跌"

    return relation


def triple_relation(
    left, mid, right, use_right=False
) -> tuple[Optional[Shape], tuple[Direction, Direction]]:
    """
    三棵缠论k线的所有关系#, 允许逆序包含存在。
    顶分型: 中间高点为三棵最高点。
    底分型: 中间低点为三棵最低点。
    上升分型: 高点从左至右依次升高
    下降分型: 低点从左至右依次降低
    喇叭口型: 高低点从左至右依次更高更低

    """
    if any((left == mid, mid == right, left == right)):
        raise ChanException("相同对象无法比较")

    shape = None
    lm = double_relation(left, mid)
    mr = double_relation(mid, right)
    # lr = double_relation(left, right)

    if lm in (Direction.Up, Direction.JumpUp):
        # 涨
        if mr in (Direction.Up, Direction.JumpUp):
            # 涨
            shape = Shape.S
        if mr in (Direction.Down, Direction.JumpDown):
            # 跌
            shape = Shape.G
        if mr is Direction.Left:
            # 顺序包含
            print("顺序包含 mr")
            raise ChanException("顺序包含 mr")
        if mr is Direction.Right and use_right:
            # 逆序包含
            shape = Shape.S

    if lm in (Direction.Down, Direction.JumpDown):
        # 跌
        if mr in (Direction.Up, Direction.JumpUp):
            # 涨
            shape = Shape.D
        if mr in (Direction.Down, Direction.JumpDown):
            # 跌
            shape = Shape.X
        if mr is Direction.Left:
            # 顺序包含
            print("顺序包含 mr")
            raise ChanException("顺序包含 mr")
        if mr is Direction.Right and use_right:
            # 逆序包含
            shape = Shape.X

    if lm is Direction.Left:
        # 顺序包含
        print("顺序包含 lm")
        raise ChanException("顺序包含 lm")

    if lm is Direction.Right and use_right:
        # 逆序包含
        if mr in (Direction.Up, Direction.JumpUp):
            # 涨
            shape = Shape.D
        if mr in (Direction.Down, Direction.JumpDown):
            # 跌
            shape = Shape.G
        if mr is Direction.Left:
            # 顺序包含
            print("顺序包含 mr")
            raise ChanException("顺序包含 mr")
        if mr is Direction.Right and use_right:
            # 逆序包含
            shape = Shape.T  # 喇叭口型
    return shape, (lm, mr)


def double_scope(left, right) -> tuple[bool, Optional["Pillar"]]:
    """
    计算重叠范围
    """
    assert left.low < left.high
    assert right.low < right.high

    if left.low < right.high <= left.high:
        # 向下
        return True, Pillar(right.high, left.low)
    if left.low <= right.low < left.high:
        # 向上
        return True, Pillar(left.high, right.low)
    if left.low <= right.low and left.high >= right.high:
        return True, Pillar(right.high, right.low)
    if left.low >= right.low and left.high <= right.high:
        return True, Pillar(left.high, left.low)

    return False, None


def triple_scope(left, mid, right) -> tuple[bool, Optional["Pillar"]]:
    b, p = double_scope(left, mid)
    if b:
        return double_scope(p, right)
    return False, None


class Observer(metaclass=ABCMeta):
    """观察者的基类"""

    CAN = False
    TIME = 0.2
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    @abstractmethod
    def update(self, observable: "Observable", **kwords: Any):
        pass


class Observable(object):
    """被观察者的基类"""

    __slots__ = ("__observers",)

    def __init__(self):
        self.__observers = []

    # 添加观察者
    def attach(self, observer: Observer):
        self.__observers.append(observer)

    # 删除观察者
    def detach(self, observer: Observer):
        self.__observers.remove(observer)

    # 内容或状态变化时通知所有的观察者
    def notify(self, **kwords):
        if Observer.CAN:
            for o in self.__observers:
                o.update(self, **kwords)


class ObservableList(Observable):
    def __init__(self, arr: List):
        super().__init__()
        self._list = list(arr)

    def __getitem__(self, index):
        return self._list[index]

    def __len__(self):
        return len(self._list)

    def __setitem__(self, index, value):
        old_value = self._list[index]
        self._list[index] = value
        self.notify(name="setitem", objs=self._list, index=index, value=value)

    def __delitem__(self, index):
        raise ChanException
        value = self._list.pop(index)
        self.notify(name="delitem", objs=self._list, index=index, value=value)

    def append(self, value):
        self._list.append(value)
        self.notify(name="append", objs=self._list, value=value)

    def extend(self, values):
        raise ChanException
        self._list.extend(values)
        self.notify(name="extend", objs=self._list, value=values)

    def insert(self, index, value):
        self._list.insert(index, value)
        self.notify(name="insert", objs=self._list, index=index, value=value)

    def pop(self, index=-1):
        value = self._list.pop(index)
        self.notify(name="pop", objs=self._list, index=index, value=value)
        return value

    def remove(self, value):
        raise ChanException
        self._list.remove(value)
        self.notify(name="remove", objs=self._list, value=value)


class TVShapeID(object):
    """
    charting_library shape ID 管理
    """

    IDS = set()
    __slots__ = "__shape_id"

    def __init__(self):
        super().__init__()
        s = TVShapeID.get(6)
        while s in TVShapeID.IDS:
            s = TVShapeID.get(6)
        TVShapeID.IDS.add(s)
        self.__shape_id: str = s

    @property
    def shape_id(self) -> str:
        return self.__shape_id

    @shape_id.setter
    def shape_id(self, value: str):
        TVShapeID.IDS.remove(self.__shape_id)
        self.__shape_id = value
        TVShapeID.IDS.add(value)

    @staticmethod
    def get(size: int):
        return "".join(
            [
                choice("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                for _ in range(size)
            ]
        )


class BaseChaoObject(Observable):
    """ """

    __slots__ = "cache", "elements", "done", "index", "pre", "__shape_id", "__pillar"
    FAST = 12
    SLOW = 26
    SIGNAL = 9

    def __init__(self, index=0):
        super().__init__()
        self.__shape_id = TVShapeID()
        self.__pillar = Pillar(0.0, 0.0)
        self.cache = dict()
        self.elements = []
        self.done = False
        self.index = index
        self.pre: Optional[Union["RawBar", "NewBar", "Bi", "Duan"]] = None

    @property
    def high(self) -> float:
        return self.__pillar.high

    @high.setter
    def high(self, value: float):
        self.__pillar.high = value

    @property
    def low(self) -> float:
        return self.__pillar.low

    @low.setter
    def low(self, value: float):
        self.__pillar.low = value

    @property
    def macd(self) -> float:
        return sum(abs(bar.macd) for bar in self.elements)

    @property
    def shape_id(self) -> str:
        return self.__shape_id.shape_id

    @shape_id.setter
    def shape_id(self, str6id: str):
        self.__shape_id.shape_id = str6id

    @classmethod
    def last(cls) -> Optional[Union["RawBar", "NewBar", "Bi", "Duan"]]:
        return cls.OBJS[-1] if cls.OBJS else None


class RawBar(BaseChaoObject, Observer):
    """
    原始K线对象

    """

    OBJS: List["RawBar"] = []
    PATCHS: Dict[int, Pillar] = dict()

    CMD_APPEND = "append"

    __slots__ = (
        "open",
        "close",
        "volume",
        "dts",
        "lv",
        "start_include",
        "belong_include",
        "shape",
    )

    def __init__(
        self, dt: datetime, o: float, h: float, l: float, c: float, v: float, i: int
    ):
        if RawBar.OBJS:
            i = RawBar.last().index + 1
        super().__init__(index=i)

        self.dt = dt
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        if pillar := RawBar.PATCHS.get(int(dt.timestamp())):
            self.high = pillar.high
            self.low = pillar.low

        self.dts = [
            self.dt,
        ]
        self.lv = self.volume  # 最新成交量，用于Tick或频繁获取最新数据时对于相同时间戳的成交量计算“真实成交量可靠性”
        self.start_include: bool = False  # 起始包含位
        self.belong_include: int = -1  # 所属包含
        self.shape: Optional[Shape] = None

        self.elements = None
        RawBar.OBJS.append(self)
        self.attach(self)
        self.notify(cmd=RawBar.CMD_APPEND)

    def __str__(self):
        return f"{self.__class__.__name__}({self.dt}, {self.high}, {self.low})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dt}, {self.high}, {self.low})"

    def update(self, observer: "Observer", **kwords: Any):
        cmd = kwords.get("cmd")
        return

        if cmd in (RawBar.CMD_APPEND,):
            message = {
                "type": "realtime",
                "timestamp": self.dt.isoformat(),
                "open": self.open,
                "close": self.close,
                "high": self.high,
                "low": self.low,
                "volume": self.volume,
            }
            future = asyncio.run_coroutine_threadsafe(
                Observer.queue.put(message), Observer.loop
            )
            try:
                future.result()  # 确保任务已添加到队列
            except Exception as e:
                print(f"Error adding task to queue: {e}")

    def __bytes__(self):
        return struct.pack(
            ">6d",
            int(self.dt.timestamp()),
            round(self.open, 8),
            round(self.high, 8),
            round(self.low, 8),
            round(self.close, 8),
            round(self.volume, 8),
        )

    @classmethod
    def from_bytes(cls, buf: bytes):
        timestamp, open, high, low, close, vol = struct.unpack(
            ">6d", buf[: struct.calcsize(">6d")]
        )
        return cls(
            dt=datetime.fromtimestamp(timestamp),
            o=open,
            h=high,
            l=low,
            c=close,
            v=vol,
            i=0,
        )

    def to_new_bar(self, pre: Optional["NewBar"]) -> "NewBar":
        return NewBar(
            dt=self.dt,
            high=self.high,
            low=self.low,
            elements=[
                self,
            ],
            pre=pre,
        )

    @property
    def macd(self) -> float:
        return self.cache[
            f"macd_{BaseChaoObject.FAST}_{BaseChaoObject.SLOW}_{BaseChaoObject.SIGNAL}"
        ]

    @property
    def ampl(self) -> float:
        """涨跌幅"""
        return (self.open - self.close) / self.open

    @property
    def direction(self) -> Direction:
        return Direction.Up if self.open < self.close else Direction.Down


class NewBar(BaseChaoObject, Observer):
    """
    缠论 K线
    """

    CMD_APPEND = "append"

    OBJS: List["NewBar"] = []

    __slots__ = "__shape", "relation", "jump", "speck", "dt", "direction"

    def __init__(
        self,
        dt: datetime,
        high: float,
        low: float,
        elements: List[RawBar],
        pre: Optional["NewBar"] = None,
    ):
        super().__init__()
        self.__shape: Optional[Shape] = None
        self.relation: Optional[Direction] = None  # 与前一个关系
        self.jump: bool = False  # 与前一个是否是跳空
        self.speck: Optional[float] = None  # 分型高低点

        self.dt = dt
        self.high = high
        self.low = low
        self.elements: List[RawBar] = elements
        # self.pre = pre
        self.direction = self.elements[
            0
        ].direction  # if self.elements else Direction.Up
        if pre is not None:
            relation = double_relation(pre, self)
            assert relation not in (Direction.Left, Direction.Right)
            self.index = pre.index + 1
            if relation in (Direction.JumpUp, Direction.JumpDown):
                self.jump = True
            self.relation = relation
            self.direction = (
                Direction.Up
                if relation in (Direction.JumpUp, Direction.Up)
                else Direction.Down
            )
        NewBar.OBJS.append(self)
        self.attach(self)
        self.notify(cmd=NewBar.CMD_APPEND)

    def update(self, observable: "Observable", **kwords: Any):
        cmd = kwords.get("cmd")
        # https://www.tradingview.com/charting-library-docs/v26/api/interfaces/Charting_Library.CreateShapeOptions/
        point = {"time": int(self.dt.timestamp())}
        options = {
            "shape": "arrow_up" if self.direction is Direction.Up else "arrow_down",
            "text": str(self.index),
        }
        if cmd in (NewBar.CMD_APPEND,):
            message = {
                "type": "realtime",
                "timestamp": self.dt.isoformat(),
                "open": self.open,
                "close": self.close,
                "high": self.high,
                "low": self.low,
                "volume": self.volume,
                "shape": {"point": point, "options": options, "id": self.shape_id},
            }
            future = asyncio.run_coroutine_threadsafe(
                Observer.queue.put(message), Observer.loop
            )
            try:
                future.result()  # 确保任务已添加到队列
            except Exception as e:
                print(f"Error adding task to queue: {e}")

    @classmethod
    def get_last_fx(cls) -> Optional["FenXing"]:
        try:
            left, mid, right = NewBar.OBJS[-3:]
        except ValueError:
            return

        left, mid, right = NewBar.OBJS[-3:]
        shape, relations = triple_relation(left, mid, right)
        mid.shape = shape

        if shape is Shape.G:
            mid.speck = mid.high
            right.speck = right.low

        if shape is Shape.D:
            mid.speck = mid.low
            right.speck = right.high

        if shape is Shape.S:
            right.speck = right.high
            right.shape = Shape.S
            mid.speck = mid.high

        if shape is Shape.X:
            right.speck = right.low
            right.shape = Shape.X
            mid.speck = mid.low

        return FenXing(left, mid, right)

    def __str__(self):
        return f"{self.__class__.__name__}({self.index}, {self.dt}, {self.high}, {self.low}, {self.shape})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.index}, {self.dt}, {self.high}, {self.low}, {self.shape})"

    def _to_raw_bar(self) -> RawBar:
        return RawBar(
            dt=self.dt,
            o=self.open,
            h=self.high,
            l=self.low,
            c=self.close,
            v=self.volume,
            i=self.index,
        )

    def merge(self, next_raw_bar: "RawBar") -> Optional["NewBar"]:
        """
        去除包含关系
        :param next_raw_bar :
        :return: 存在包含关系返回 None, 否则返回下一个 NewBar
        """
        assert next_raw_bar.index - 1 == self.elements[-1].index
        relation = double_relation(self, next_raw_bar)
        if relation in (Direction.Left, Direction.Right):
            # 合并
            if self.direction is Direction.Up:
                self.high = max(self.high, next_raw_bar.high)
                self.low = max(self.low, next_raw_bar.low)
            else:
                self.high = min(self.high, next_raw_bar.high)
                self.low = min(self.low, next_raw_bar.low)

            assert next_raw_bar.index - 1 == self.elements[-1].index
            self.notify(cmd=NewBar.CMD_APPEND)

            self.elements.append(next_raw_bar)
            return None
        self.done = True
        return next_raw_bar.to_new_bar(self)

    @property
    def shape(self) -> Optional[Shape]:
        return self.__shape

    @shape.setter
    def shape(self, shape: Shape):
        self.__shape = shape
        if shape is None:
            self.speck = None
        if shape is Shape.G:
            self.speck = self.high
        if shape is Shape.S:
            self.speck = self.high
        if shape is Shape.D:
            self.speck = self.low
        if shape is Shape.X:
            self.speck = self.low

    @property
    def volume(self) -> float:
        """
        :return: 总计成交量
        """
        return sum([raw.volume for raw in self.elements])

    @property
    def open(self) -> float:
        return self.high if self.direction == Direction.Down else self.low

    @property
    def close(self) -> float:
        return self.low if self.direction == Direction.Down else self.high


class FenXing(BaseChaoObject):
    """
    缠论 分型
    """

    __slots__ = "left", "mid", "right", "__shape", "__speck"
    OBJS: List["FenXing"] = []

    def __init__(self, left: NewBar, mid: NewBar, right: NewBar, index: int = 0):
        super().__init__()
        self.left = left
        self.mid = mid
        self.right = right
        self.index = index

        self.__shape = mid.shape
        self.__speck = mid.speck
        self.elements = [left, mid, right]

    def next_new_bar(self, next_new_bar: NewBar) -> None:
        assert next_new_bar.index - 1 == self.right.index
        self.done = True

    @property
    def dt(self) -> datetime:
        return self.mid.dt

    @property
    def shape(self) -> Shape:
        return self.__shape

    @property
    def speck(self) -> float:
        return self.__speck

    @property
    def high(self) -> float:
        return max(self.left.high, self.mid.high)

    @property
    def low(self) -> float:
        return min(self.left.low, self.mid.low)

    def __str__(self):
        return f"FenXing({self.shape}, {self.speck}, {self.dt})"

    def __repr__(self):
        return f"FenXing({self.shape}, {self.speck}, {self.dt})"

    @staticmethod
    def append(fxs, fx):
        if fxs and fxs[-1].shape is fx.shape:
            raise ChanException("分型相同无法添加", fxs[-1], fx)
        i = 0
        if fxs:
            i = fxs[-1].index + 1
        fx.index = i
        fxs.append(fx)

    @staticmethod
    def pop(fxs, fx):
        if fxs and fxs[-1] is not fx:
            raise ChanException("分型相同无法删除", fxs[-1], fx)
        return fxs.pop()


class Bi(BaseChaoObject, Observer):
    """
    缠论笔
    """

    OBJS: List["Bi"] = []
    FAKE: "Bi" = None  # 最新未完成笔

    BI_LENGTH = 5  # 成BI最低长度
    BI_JUMP = True  # 跳空是否是一个NewBar
    BI_FENGXING = False  # True: 一笔起始分型高低包含整个笔对象则不成笔, False: 只判断分型中间数据是否包含
    CMD_APPEND = "append"
    CMD_MODIFY = "modify"
    CMD_REMOVE = "remove"

    __slots__ = "direction", "__start", "__end", "flag"

    def __init__(
        self,
        pre: Optional["Self"],
        start: FenXing,
        end: Union[FenXing, NewBar],
        elements: Optional[List[NewBar]],
        flag: bool = True,
    ):
        super().__init__()
        if start.shape is Shape.G:
            self.direction = Direction.Down
            self.high = start.speck
            self.low = end.low
        elif start.shape is Shape.D:
            self.direction = Direction.Up
            self.high = end.high
            self.low = start.speck
        else:
            raise ChanException(start.shape, end.shape)
        for i in range(1, len(self.elements)):
            assert self.elements[i - 1].index + 1 == self.elements[i].index, (
                self.elements[i - 1].index,
                self.elements[i].index,
            )
        if pre is not None:
            assert pre.end is start, (pre.end, start)
            self.index = pre.index + 1
        self.pre = pre
        self.__start = start
        self.__end = end
        self.elements = elements
        """if Bi.OBJS:
            last = Bi.OBJS[-1]
            assert last.elements[-1] is elements[0], (
                last.elements[-1],
                elements[0],
            )"""
        self.flag = flag

        self.attach(self)  # 自我观察
        if self.flag:
            Bi.OBJS.append(self)
            self.notify(cmd=Bi.CMD_APPEND)

    @staticmethod
    def append(bis, bi):
        if bis and bis[-1].end is not bi.start:
            raise TypeError("笔连续性错误")
        i = 0
        if bis:
            i = bis[-1].index + 1
        bi.index = i
        bis.append(bi)
        bi.notify(cmd=Bi.CMD_APPEND)

    @staticmethod
    def pop(bis, fx):
        if bis:
            if bis[-1].end is fx:
                bi = bis.pop()
                bi.notify(cmd=Bi.CMD_REMOVE)
                return
            else:
                raise ValueError("最后一笔终点错误", fx, bis[-1].end)

    def __str__(self):
        return f"Bi({self.direction}, {colored(self.start.dt, 'green')}, {self.start.speck}, {colored(self.end.dt, 'green')}, {self.end.speck}, {self.index}, {self.elements[-1]})"

    def __repr__(self):
        return f"Bi({self.direction}, {colored(self.start.dt, 'green')}, {self.start.speck}, {colored(self.end.dt, 'green')}, {self.end.speck}, {self.index}, {self.elements[-1]})"

    def update(self, observable: "Observable", **kwords: Any):
        # 实现 自我观察
        cmd = kwords.get("cmd")
        points = [
            {"time": int(self.start.dt.timestamp()), "price": self.start.speck},
            {"time": int(self.elements[-1].dt.timestamp()), "price": self.end.speck},
        ]
        options = {
            "shape": "trend_line",
            "linecolor": "#FF0000",
            "linewidth": 1,
            "lock": True,
        }
        message = {
            "type": "shape",
            "cmd": cmd,
            "name": "trend_line",
            "id": self.shape_id,
            "points": points,
            "options": options,
        }
        if cmd in (Bi.CMD_APPEND, Bi.CMD_REMOVE, Bi.CMD_MODIFY):
            # 后端实现 增 删 改
            future = asyncio.run_coroutine_threadsafe(
                Observer.queue.put(message), Observer.loop
            )
            try:
                future.result()  # 确保任务已添加到队列
            except Exception as e:
                print(f"Error adding task to queue: {e}")

    @property
    def length(self) -> int:
        return Bi.calc_length(self.elements)

    @property
    def start(self) -> FenXing:
        return self.__start

    @start.setter
    def start(self, start: FenXing):
        """

        :param start:
        :return:
        """
        if start is None:
            self.notify(cmd=Bi.CMD_REMOVE)
            self.__start = start
            self.elements = None
            assert Bi.OBJS[-1] is self, Bi.OBJS[-1]
            if self.flag:
                Bi.OBJS.remove(self)
            return
        assert start.shape in (Shape.G, Shape.D)

        self.__start = start
        if self.direction is Direction.Down:
            assert start.shape is Shape.G
            self.high = start.speck

        if self.direction is Direction.Up:
            assert start.shape is Shape.D
            self.low = start.speck

        if self not in Bi.OBJS:
            Bi.OBJS.append(self)
            self.notify(cmd=Bi.CMD_APPEND)
        else:
            self.notify(cmd=Bi.CMD_MODIFY)

    @property
    def end(self) -> FenXing:
        return self.__end

    @end.setter
    def end(self, end: Union[FenXing, NewBar]):
        old = self.__end
        self.__end = end
        tag = True
        if self.direction is Direction.Down:
            if old.low == end.low:
                tag = False
            self.low = min(self.low, end.low)
        if self.direction is Direction.Up:
            if old.high == end.high:
                tag = False
            self.high = max(self.high, end.high)
        if tag:
            self.notify(cmd=Bi.CMD_MODIFY)

    @property
    def real_high(self) -> NewBar:
        return max(self.elements, key=lambda x: x.high) if self.elements else None

    @property
    def real_low(self) -> NewBar:
        return min(self.elements, key=lambda x: x.low) if self.elements else None

    @property
    def relation(self) -> bool:
        if Bi.BI_FENGXING:
            start = self.start
        else:
            start = self.start.mid

        if self.direction is Direction.Down:
            return double_relation(start, self.end) in (
                Direction.Down,
                Direction.JumpDown,
            )
        return double_relation(start, self.end) in (Direction.Up, Direction.JumpUp)

    @staticmethod
    def calc_length(elements) -> int:
        size = 1
        # elements = self.elements
        for i in range(1, len(elements)):
            left = elements[i - 1]
            right = elements[i]
            assert left.index + 1 == right.index, (
                left.index,
                right.index,
            )
            relation = double_relation(left, right)
            if Bi.BI_JUMP and relation in (Direction.JumpUp, Direction.JumpDown):
                size += 1
            size += 1
        if not Bi.BI_JUMP:
            assert size == len(elements)
        if Bi.BI_JUMP:
            return size
        return len(elements)

    def __append_and_calc(self, new_bar: NewBar):
        assert self.elements[-1].index + 1 == new_bar.index, (
            new_bar,
            self.elements[-1],
        )
        self.elements.append(new_bar)
        if self.direction is Direction.Down:
            old = self.low
            if old > new_bar.low:
                self.__end = new_bar
            self.low = min(self.low, new_bar.low)
            self.notify(cmd=Bi.CMD_MODIFY)
            if self.real_high is not self.start.mid:
                dp("不是真顶", self)

        if self.direction is Direction.Up:
            old = self.high
            if old < new_bar.high:
                self.__end = new_bar
            self.high = max(self.high, new_bar.high)
            self.notify(cmd=Bi.CMD_MODIFY)
            if self.real_low is not self.start.mid:
                dp("不是真底", self)

    def check(self) -> bool:
        if len(self.elements) >= 5:
            assert self.start.mid is self.elements[0]
            assert self.end.mid is self.elements[-1]
            if (
                self.direction is Direction.Down
                and self.start.mid is self.real_high
                and self.end.mid is self.real_low
            ):
                return True
            if (
                self.direction is Direction.Up
                and self.start.mid is self.real_low
                and self.end.mid is self.real_high
            ):
                return True
        return False

    @classmethod
    def calc_fake(cls):
        ...
        last = cls.FAKE
        if last is not None:
            last.notify(cmd=Bi.CMD_REMOVE)

        if FenXing.OBJS:
            start = FenXing.OBJS[-1]
            elememts = NewBar.OBJS[start.mid.index :]
            low = min(elememts, key=lambda x: x.low)
            high = max(elememts, key=lambda x: x.high)
            pre = cls.last()
            if start.shape is Shape.G:
                bi = Bi(pre, start, low, elememts, flag=False)
            else:
                bi = Bi(pre, start, high, elememts, flag=False)
            cls.FAKE = bi
            bi.notify(cmd=Bi.CMD_APPEND)

    @staticmethod
    def analyzer(
        fx: FenXing,
        fxs: List[FenXing],
        bis: List["Bi"],
        cklines: List[NewBar],
    ):
        last = fxs[-1] if fxs else None
        left, mid, right = fx.left, fx.mid, fx.right
        if Bi.FAKE:
            Bi.FAKE.notify(cmd=Bi.CMD_REMOVE)
            Bi.FAKE = None
        if last is None:
            if mid.shape in (Shape.G, Shape.D):
                fxs.append(fx)
            return

        if last.mid.dt > fx.mid.dt:
            raise TypeError("时序错误")

        if last.shape is Shape.G and fx.shape is Shape.D:
            bi = Bi(
                None,
                last,
                fx,
                cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                flag=False,
            )
            if bi.length > 4:
                if bi.real_high is not last.mid:
                    # print("不是真顶")
                    top = bi.real_high
                    new = FenXing(
                        cklines[cklines.index(top) - 1],
                        top,
                        cklines[cklines.index(top) + 1],
                    )
                    assert new.shape is Shape.G, new
                    Bi.analyzer(new, fxs, bis, cklines)  # 处理新顶
                    Bi.analyzer(fx, fxs, bis, cklines)  # 再处理当前底

                    return
                flag = bi.relation
                if flag and fx.mid is bi.real_low:
                    FenXing.append(fxs, fx)
                    Bi.append(bis, bi)

                else:
                    ...
                    # 2024 05 21 修正
                    _cklines = cklines[last.mid.index :]
                    _fx, _bi = Bi.analysis_one(_cklines)

                    if _bi and len(fxs) > 2:
                        nb = Bi(
                            None,
                            fxs[-3],
                            _bi.start,
                            cklines[fxs[-3].mid.index : _bi.start.mid.index + 1],
                            flag=False,
                        )
                        _bi.notify(cmd=Bi.CMD_REMOVE)
                        if not nb.check():
                            return
                        print(_bi)
                        tmp = fxs.pop()
                        assert tmp is last
                        bi = Bi.pop(bis, tmp)

            else:
                ...
                # GD
                if right.high > last.speck:
                    tmp = fxs.pop()
                    assert tmp is last
                    bi = Bi.pop(bis, tmp)

        elif last.shape is Shape.D and fx.shape is Shape.G:
            bi = Bi(
                None,
                last,
                fx,
                cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                flag=False,
            )
            if bi.length > 4:
                if bi.real_low is not last.mid:
                    # print("不是真底")
                    bottom = bi.real_low
                    new = FenXing(
                        cklines[cklines.index(bottom) - 1],
                        bottom,
                        cklines[cklines.index(bottom) + 1],
                    )
                    assert new.shape is Shape.D, new
                    Bi.analyzer(new, fxs, bis, cklines)  # 处理新底
                    Bi.analyzer(fx, fxs, bis, cklines)  # 再处理当前顶

                    return
                flag = bi.relation
                if flag and fx.mid is bi.real_high:
                    FenXing.append(fxs, fx)
                    Bi.append(bis, bi)

                else:
                    ...
                    # 2024 05 21 修正
                    _cklines = cklines[last.mid.index :]
                    _fx, _bi = Bi.analysis_one(_cklines)

                    if _bi and len(fxs) > 2:
                        nb = Bi(
                            None,
                            fxs[-3],
                            _bi.start,
                            cklines[fxs[-3].mid.index : _bi.start.mid.index + 1],
                            False,
                        )
                        _bi.notify(cmd=Bi.CMD_REMOVE)
                        if not nb.check():
                            return
                        print(_bi)
                        tmp = fxs.pop()
                        assert tmp is last
                        bi = Bi.pop(bis, tmp)

            else:
                ...
                # DG
                if right.low < last.speck:
                    tmp = fxs.pop()
                    assert tmp is last
                    bi = Bi.pop(bis, tmp)

        elif last.shape is Shape.G and fx.shape is Shape.S:
            if last.speck < right.high:
                tmp = fxs.pop()
                bi = Bi.pop(bis, tmp)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.D
                    bottom = min(
                        cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                        key=lambda o: o.low,
                    )
                    assert bottom.shape is Shape.D
                    if last.speck > bottom.low:
                        tmp = fxs.pop()
                        bi = Bi.pop(bis, tmp)

                        new = FenXing(
                            cklines[cklines.index(bottom) - 1],
                            bottom,
                            cklines[cklines.index(bottom) + 1],
                        )
                        assert new.shape is Shape.D, new
                        Bi.analyzer(new, fxs, bis, cklines)  # 处理新底
                        # print("GS修正")

        elif last.shape is Shape.D and fx.shape is Shape.X:
            if last.speck > right.low:
                tmp = fxs.pop()
                bi = Bi.pop(bis, tmp)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.G
                    top = max(
                        cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                        key=lambda o: o.high,
                    )
                    assert top.shape is Shape.G
                    if last.speck < top.high:
                        tmp = fxs.pop()
                        bi = Bi.pop(bis, tmp)

                        new = FenXing(
                            cklines[cklines.index(top) - 1],
                            top,
                            cklines[cklines.index(top) + 1],
                        )
                        assert new.shape is Shape.G, new
                        Bi.analyzer(new, fxs, bis, cklines)  # 处理新顶
                        # print("DX修正")

        elif last.shape is Shape.G and fx.shape is Shape.G:
            if last.speck < fx.speck:
                tmp = fxs.pop()
                bi = Bi.pop(bis, tmp)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.D
                    bottom = min(
                        cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                        key=lambda o: o.low,
                    )
                    assert bottom.shape is Shape.D
                    if last.speck > bottom.low:
                        tmp = fxs.pop()
                        bi = Bi.pop(bis, tmp)

                        new = FenXing(
                            cklines[cklines.index(bottom) - 1],
                            bottom,
                            cklines[cklines.index(bottom) + 1],
                        )
                        assert new.shape is Shape.D, new
                        Bi.analyzer(new, fxs, bis, cklines)  # 处理新底
                        Bi.analyzer(fx, fxs, bis, cklines)  # 再处理当前顶
                        # print("GG修正")
                        return

                if not fxs:
                    FenXing.append(fxs, fx)
                    return
                bi = Bi(
                    None,
                    fxs[-1],
                    fx,
                    cklines[cklines.index(fxs[-1].mid) : cklines.index(fx.mid) + 1],
                    flag=False,
                )
                FenXing.append(fxs, fx)
                Bi.append(bis, bi)

        elif last.shape is Shape.D and fx.shape is Shape.D:
            if last.speck > fx.speck:
                tmp = fxs.pop()
                bi = Bi.pop(bis, tmp)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.G
                    top = max(
                        cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1],
                        key=lambda o: o.high,
                    )
                    assert top.shape is Shape.G
                    if last.speck < top.high:
                        tmp = fxs.pop()
                        bi = Bi.pop(bis, tmp)

                        new = FenXing(
                            cklines[cklines.index(top) - 1],
                            top,
                            cklines[cklines.index(top) + 1],
                        )
                        assert new.shape is Shape.G, new
                        Bi.analyzer(new, fxs, bis, cklines)  # 处理新顶
                        Bi.analyzer(fx, fxs, bis, cklines)  # 再处理当前底
                        # print("DD修正")
                        return

                if not fxs:
                    FenXing.append(fxs, fx)
                    return
                bi = Bi(
                    None,
                    fxs[-1],
                    fx,
                    cklines[cklines.index(fxs[-1].mid) : cklines.index(fx.mid) + 1],
                    flag=False,
                )
                FenXing.append(fxs, fx)
                Bi.append(bis, bi)

        elif last.shape is Shape.G and fx.shape is Shape.X:
            ...

        elif last.shape is Shape.D and fx.shape is Shape.S:
            ...

        else:
            raise ValueError(last.shape, fx.shape)

    def analysis_one(cklines: List[NewBar]) -> tuple[Optional[FenXing], Optional["Bi"]]:
        try:
            cklines[2]
        except IndexError:
            return None, None
        bis = []
        fxs = []
        fx = None
        size = len(cklines)
        for i in range(1, size - 2):
            left, mid, right = cklines[i - 1], cklines[i], cklines[i + 1]
            fx = FenXing(left, mid, right)
            Bi.analyzer(fx, fxs, bis, cklines)
            if bis:
                return fx, bis[0]
        if bis:
            return fx, bis[0]

        return None, None


class Duan(BaseChaoObject):
    OBJS: List["Duan"] = []

    # __slots__ =
    def __init__(self):
        super().__init__()


class BaseAnalyzer:
    def __init__(self, symbol: str, freq: int):
        self.__symbol = symbol
        self.__freq = freq
        RawBar.OBJS = []
        NewBar.OBJS = []
        FenXing.OBJS = []
        Bi.OBJS = []
        Duan.OBJS = []
        self._raws: List[RawBar] = RawBar.OBJS  # 原始K线列表
        self._news: List[NewBar] = NewBar.OBJS  # 去除包含关系K线列表
        self._fxs: List[FenXing] = FenXing.OBJS  # 分型列表
        self._bis: List[Bi] = Bi.OBJS  # 笔
        self._duans: List[Duan] = Duan.OBJS

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def freq(self) -> int:
        return self.__freq

    def xd_zs_bc(self): ...

    def push(
        self,
        bar: RawBar,
        fast_period: int = BaseChaoObject.FAST,
        slow_period: int = BaseChaoObject.SLOW,
        signal_period: int = BaseChaoObject.SIGNAL,
    ):
        last = self._news[-1] if self._news else None
        news = self._news
        new_bar = None
        if last is None:
            bar.to_new_bar(None)
        else:
            new_bar: Optional[NewBar] = last.merge(bar)

        klines = news
        if len(klines) == 1:
            ema_slow = klines[-1].close
            ema_fast = klines[-1].close
        else:
            ema_slow = (
                2 * klines[-1].close
                + klines[-2].cache[f"ema_{slow_period}"] * (slow_period - 1)
            ) / (slow_period + 1)
            ema_fast = (
                2 * klines[-1].close
                + klines[-2].cache[f"ema_{fast_period}"] * (fast_period - 1)
            ) / (fast_period + 1)

        klines[-1].cache[f"ema_{slow_period}"] = ema_slow
        klines[-1].cache[f"ema_{fast_period}"] = ema_fast
        DIF = ema_fast - ema_slow
        klines[-1].cache[f"dif_{fast_period}_{slow_period}_{signal_period}"] = DIF

        if len(klines) == 1:
            dea = DIF
        else:
            dea = (
                2 * DIF
                + klines[-2].cache[f"dea_{fast_period}_{slow_period}_{signal_period}"]
                * (signal_period - 1)
            ) / (signal_period + 1)

        klines[-1].cache[f"dea_{fast_period}_{slow_period}_{signal_period}"] = dea
        macd = (DIF - dea) * 2
        klines[-1].cache[f"macd_{fast_period}_{slow_period}_{signal_period}"] = macd

        fx: Optional[FenXing] = NewBar.get_last_fx()
        if fx is not None:
            # return Bi.analyzer(fx, FenXing.OBJS, Bi.OBJS, NewBar.OBJS)
            if self._bis:
                Bi.analyzer(fx, FenXing.OBJS, Bi.OBJS, NewBar.OBJS)
                # Bi.last().append_new_bar(new_bar)
                # if fx.shape in (Shape.G, Shape.D):
                #    Bi.last().append_fx(fx)
            else:
                Bi.analyzer(fx, FenXing.OBJS, Bi.OBJS, NewBar.OBJS)
                # if fx.shape in (Shape.G, Shape.D):
                #    Bi(None, fx, fx.right, [fx.mid, fx.right])
        Bi.calc_fake()


class CZSCAnalyzer:
    def __init__(self, symbol: str, freq: int, freqs: List[int] = None):
        if freqs is None:
            freqs = [freq]
        else:
            freqs.append(freq)
            freqs = list(set(freqs))
        self.symbol = symbol
        self.freq = freq
        self.freqs = freqs

        self._analyzeies = dict()
        self.__analyzer = BaseAnalyzer(symbol, freq)
        self.raws = RawBar.OBJS

    @property
    def news(self):
        return self.__analyzer._news

    @final
    def step(
        self,
        dt: datetime | int | str,
        open: float | str,
        high: float | str,
        low: float | str,
        close: float | str,
        volume: float | str,
    ):
        if type(dt) is datetime:
            ...
        elif isinstance(dt, str):
            dt: datetime = datetime.fromtimestamp(int(dt))
        elif isinstance(dt, int):
            dt: datetime = datetime.fromtimestamp(dt)
        else:
            raise ChanException("类型不支持", type(dt))
        open = float(open)
        high = float(high)
        low = float(low)
        close = float(close)
        volume = float(volume)

        index = 0

        last = RawBar(
            dt=dt,
            o=open,
            h=high,
            l=low,
            c=close,
            v=volume,
            i=index,
        )
        self.push(last)

    def push(self, k: RawBar):
        if Observer.CAN:
            time.sleep(Observer.TIME)
        try:
            self.__analyzer.push(k)
        except Exception as e:
            # self.__analyzer.toCharts()
            # with open(f"{self.symbol}-{int(self._bars[0].dt.timestamp())}-{int(self._bars[-1].dt.timestamp())}.dat", "wb") as f:
            #    f.write(self.save_bytes())
            raise e

    @classmethod
    def load_bytes(cls, symbol: str, bytes_data: bytes, freq: int) -> "Self":
        size = struct.calcsize(">6d")
        obj = cls(symbol, freq)
        bytes_data = bytes_data[size * 1250 :]
        while bytes_data:
            t = bytes_data[:size]
            k = RawBar.from_bytes(t)
            obj.push(k)

            bytes_data = bytes_data[size:]
            if len(bytes_data) < size:
                break
        return obj

    def save_bytes(self) -> bytes:
        data = b""
        for k in self.raws:
            data += bytes(k)
        return data

    def save_file(self):
        with open(
            f"{self.symbol}-{self.freq}-{int(self.__analyzer._raws[0].dt.timestamp())}-{int(self.__analyzer._raws[-1].dt.timestamp())}.dat",
            "wb",
        ) as f:
            f.write(self.save_bytes())

    @classmethod
    def load_file(cls, path: str) -> "Self":
        name = Path(path).name.split(".")[0]
        symbol, freq, s, e = name.split("-")
        with open(path, "rb") as f:
            dat = f.read()
            return cls.load_bytes(symbol, dat, int(freq))


def main_load_file(path: str = "btcusd-300-1713295800-1715695500.dat"):
    return CZSCAnalyzer.load_file(path)


app = FastAPI()
# priority_queue = asyncio.PriorityQueue()
# queue = Observer.queue  # asyncio.Queue()
app.mount(
    "/charting_library",
    StaticFiles(directory="charting_library"),
    name="charting_library",
)
templates = Jinja2Templates(directory="templates")


async def process_queue():
    while True:
        message = await Observer.queue.get()
        try:
            await handle_message(message)
        except Exception as e:
            print(f"Error handling message: {e}")
            traceback.print_exc()
        finally:
            Observer.queue.task_done()


@app.on_event("startup")
async def startup_event():
    # 启动队列处理任务
    asyncio.create_task(process_queue())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["type"] == "ready":
                thread = Thread(target=main_load_file)  # 使用线程来运行main函数
                thread.start()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/{nol}/{exchange}/{symbol}", response_class=HTMLResponse)
async def home(request: Request, nol: str, exchange: str, symbol: str):
    print(dir(request))
    print(request.base_url)
    charting_library = str(
        request.url_for("charting_library", path="/charting_library.standalone.js")
    )
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="zh">
<head>
    <title>TradingView Chart with WebSocket</title>
    <meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,minimum-scale=1.0">
    <script type="text/javascript" src="$charting_library$"></script>
    <script type="text/javascript">
        const shape_ids = new Array(); // id 映射
        const socket = new WebSocket('ws://localhost:8080/ws');

        socket.onopen = () => {
            console.log("WebSocket connection established");
            socket.send(JSON.stringify({type: 'ready'}));
        };

        socket.onclose = () => {
            console.log("WebSocket connection closed");
        };

        socket.onerror = (error) => {
            console.error("WebSocket error:", error);
        };

        let datafeed = {
            onReady: (callback) => {
                console.log("[Datafeed.onReady]: Method call");
                setTimeout(() => callback({
                    supports_search: false,
                    supports_group_request: false,
                    supports_marks: false,
                    supports_timescale_marks: true,
                    supports_time: true,
                    supported_resolutions: ['1s', '1', '3', '5', '6', '12', '24', '30', '48', '64', '128', '1H', '2H', '3H', '4H', '6H', '8H', '12H', '36H', '1D', '2D', '3D', '5D', '12D', '1W'],
                }));
            },
            searchSymbols: async (
                userInput,
                exchange,
                symbolType,
                onResultReadyCallback,
            ) => {
                console.log("[Datafeed.searchSymbols]: Method call", userInput, exchange, symbolType);

            },
            resolveSymbol: async (
                symbolName,
                onSymbolResolvedCallback,
                onResolveErrorCallback,
                extension
            ) => {
                console.log("[Datafeed.resolveSymbol]: Method call", symbolName);
                //return ;
                const symbolInfo = {
                    exchange: "okex",
                    ticker: 'BTCUSD',
                    name: 'BTCUSD',
                    description: 'Bitcoin/USD',
                    type: "",
                    session: '24x7',
                    timezone: 'Asia/Shanghai',
                    minmov: 1,
                    pricescale: 100,
                    visible_plots_set: 'ohlcv',
                    has_no_volume: true,
                    has_weekly_and_monthly: false, // 周线 月线
                    //supported_resolutions: ['15S', '1', '3', '5', '6', '12', '24', '30', '48', '64', '128', '1H', '2H', '3H', '4H', '6H', '8H', '12H', '36H', '1D', '2D', '3D', '5D', '12D'],
                    volume_precision: 1,
                    data_status: 'streaming',
                    has_intraday: true,
                    intraday_multipliers: ['1', "3", "5", '15', "30", '60', "120", "240", "360", "720"],
                    has_seconds: false,
                    seconds_multipliers: ['1S',],
                    has_daily: true,
                    daily_multipliers: ['1', '3'],
                    has_ticks: true,
                    monthly_multipliers: [],
                    weekly_multipliers: [],
                };
                try {
                    onSymbolResolvedCallback(symbolInfo);
                } catch (err) {
                    onResolveErrorCallback(err.message);
                }

            },
            getBars: async (
                symbolInfo,
                resolution,
                periodParams,
                onHistoryCallback,
                onErrorCallback,
            ) => {
                const {from, to, firstDataRequest} = periodParams;
                console.log("[Datafeed.getBars]: Method call", symbolInfo, resolution, from, to, firstDataRequest);
                try {
                    onHistoryCallback([], {noData: true});

                } catch (error) {
                    console.log("[Datafeed.getBars]: Get error", error);
                    onErrorCallback(error);
                }
            },
            subscribeBars: (
                symbolInfo,
                resolution,
                onRealtimeCallback,
                subscriberUID,
                onResetCacheNeededCallback,
            ) => {
                console.log(
                    "[Datafeed.subscribeBars]: Method call with subscriberUID:",
                    symbolInfo,
                    resolution,
                    subscriberUID,
                );
                socket.onmessage = function (event) {
                    const message = JSON.parse(event.data);
                    console.info(message);
                    if (message.type === "realtime") {
                        const bar = {
                            time: new Date(message.timestamp).getTime(), // Unix timestamp in milliseconds
                            close: message.close,
                            open: message.open,
                            high: message.high,
                            low: message.low,
                            volume: message.volume,
                        };
                        onRealtimeCallback(bar);
                        //createShape(message.shape);
                    } else if (message.type === "shape") {
                        if (message.cmd === "append") {
                            addShapeToChart(message);
                        } else if (message.cmd === "remove") {
                            delShapeById(message.id)
                        } else if (message.cmd === "modify") {
                            modifyShape(message)
                        }

                    } else {
                        console.log(message);
                    }
                };
            },
            unsubscribeBars: (subscriberUID) => {
                console.log(
                    "[Datafeed.unsubscribeBars]: Method call with subscriberUID:",
                    subscriberUID,
                );
                socket.close();
            }
        };
        

        function addShapeToChart(obj) {
            if (window.tvWidget) {
                console.log(obj);
                const shape_id = window.tvWidget.chart().createMultipointShape(obj.points, obj.options);
                shape_ids [obj.id] = shape_id;
                const shape = window.tvWidget.chart().getShapeById(shape_id);
                shape.bringToFront();
                //console.log(obj.id, shape_id);

            }
        }

        function delShapeById(shapeId) {
            if (window.tvWidget) {
                const id = shape_ids[shapeId];
                const shape = window.tvWidget.chart().getShapeById(id);
                console.log(id, shape);
                window.tvWidget.chart().removeEntity(id);
                delete shape_ids[shapeId];
                console.log("del", shapeId, id);

            }
        }
        function createShape(obj) {
            if (window.tvWidget) {
                const shape_id = window.tvWidget.chart().createShape(obj.point, obj.options);
                shape_ids [obj.id] = shape_id;
                const shape = window.tvWidget.chart().getShapeById(shape_id);
                shape.bringToFront();
                
            }
        }

        function modifyShape(obj) {
            const id = shape_ids[obj.id];
            const shape = window.tvWidget.chart().getShapeById(id);
            if (shape) {
                console.log(obj);
                shape.setPoints(obj.points);
                //shape.setProperties(obj.options);
                shape.bringToFront();
                
            } else {
                console.log("Shape does not exist.");
            }
        }

        function initOnReady() {
            console.log("init widget");
            const widget = (window.tvWidget = new TradingView.widget({
                symbol: "Bitfinex:BTC/USD", // Default symbol
                interval: "5", // Default interval
                timezone: "Asia/Shanghai",
                fullscreen: true, // Displays the chart in the fullscreen mode
                container: "tv_chart_container", // Reference to an attribute of the DOM element
                datafeed: datafeed,
                library_path: "charting_library/",
                locale: "zh",
                theme: "dark",
                debug: false,
                user_id: 'public_user_id',
                client_id: 'yourserver.com',
                favorites: {
                    intervals: ["1", "3", "5"],
                    drawingTools: ["LineToolPath", "LineToolRectangle", "LineToolTrendLine"],
                },
                disabled_features: ["use_localstorage_for_settings", "header_symbol_search"],
            }));
        }

        window.addEventListener("DOMContentLoaded", initOnReady, false);

    </script>
</head>
    <body style="margin:0px;">
        <div id="tv_chart_container"></div>
    </body>
</html>

        """.replace("$charting_library$", charting_library)
    )


@app.get("/")
async def main_page(request: Request):
    Observer.CAN = True
    # Observer.loop = asyncio.get_event_loop()
    return await home(request, "local", "oke", "btcusd")


async def handle_message(message: dict):
    if message["type"] == "realtime":
        await manager.send_message(json.dumps(message))
    elif message["type"] == "shape":
        await manager.send_message(
            json.dumps(
                {
                    "type": "shape",
                    "name": message["name"],
                    "points": message["points"],
                    "id": message["id"],
                    "cmd": message["cmd"],
                    "options": message["options"],
                }
            )
        )
    elif message["type"] == "heartbeat":
        await manager.send_message(
            json.dumps(
                {
                    "type": "heartbeat",
                    "timestamp": message["timestamp"],
                }
            )
        )
    else:
        await manager.send_message(
            json.dumps({"type": "error", "message": "Unknown command type"})
        )


def synchronous_handle_message(message):
    # 向优先级队列中添加任务
    loop = asyncio.get_event_loop()
    loop.call_soon_threadsafe(Observer.queue.put_nowait, message)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


# RawBar.PATCHS[ts2int("2024-04-17 21:20:00")] = Pillar(62356, 62100)
manager = ConnectionManager()
if __name__ == "__main__":
    bit = main_load_file("btcusd-300-1713295800-1715695500.dat")
