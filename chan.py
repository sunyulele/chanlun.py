"""
# -*- coding: utf-8 -*-
# @Time    : 2024/04/28 16:45
# @Author  : YuYuKunKun
# @File    : chan.py
"""

import math
import struct
import traceback
from pathlib import Path
from typing import List, Union, Self, Literal, Optional, Tuple, final, Dict, Iterable

from dataclasses import dataclass
from datetime import datetime
from importlib import reload
from enum import Enum

import requests

try:
    from termcolor import colored
except ImportError:

    def colored(text, color="red", on_color=None, attrs=None):
        """彩色字"""
        return text


class Shape(Enum):
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
    if 1:
        _print(*args, **kwords)


def bdp(*args, **kwargs):
    if not 1:
        dp(*args, **kwargs)


def ddp(*args, **kwargs):
    if not 1:
        dp(*args, **kwargs)


def Klines(cls):
    def cklines(self, news: list["NewBar"]):
        return news[self.start.left.index : self.end.right.index]

    cls.cklines = cklines
    return cls


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


def triple_relation(left, mid, right, use_right=False) -> tuple[Optional[Shape], tuple[Direction, Direction]]:
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

class Pillar:
    def __init__(self, high: float, low: float):
        self.low = low
        self.high = high


class RawBar:
    __slots__ = "dt", "open", "high", "low", "close", "volume", "index", "cache", "done", "dts", "lv", "start_include", "belong_include"

    def __init__(self, dt: datetime, open: float, high: float, low: float, close: float, volume: float, index: int = 0):
        self.dt = dt
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.index = index
        self.cache = dict()
        self.done = False
        self.dts = [
            self.dt,
        ]
        self.lv = self.volume  # 最新成交量，用于Tick或频繁获取最新数据时对于相同时间戳的成交量计算“真实成交量可靠性”
        self.start_include: bool = False  # 起始包含位
        self.belong_include: int = -1  # 所属包含

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
        timestamp, open, high, low, close, vol = struct.unpack(">6d", buf)
        return cls(dt=datetime.fromtimestamp(timestamp), open=open, high=high, low=low, close=close, volume=vol)

    @property
    def ampl(self) -> float:
        """涨跌幅"""
        return (self.open - self.close) / self.open

    @property
    def direction(self) -> Direction:
        return Direction.Up if self.open < self.close else Direction.Down

    @property
    def new(self) -> "NewBar":
        if self.open > self.close:
            open_ = self.high
            close = self.low
        else:
            open_ = self.low
            close = self.high
        return NewBar(
            dt=self.dt,
            open=open_,
            high=self.high,
            low=self.low,
            close=close,
            index=0,
            volume=self.volume,
            elements=[
                self,
            ],
        )

    def candleDict(self):
        return {
            "dt": self.dt,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "vol": self.volume,
        }


class RawBars:
    __slots__ = "__bars", "__size", "freq", "klines"

    def __init__(self, bars: Optional[List[RawBar]] = None, freq: Optional[int] = None, merger: Optional[List] = None):
        if bars is None:
            bars = []
        self.__bars: List[RawBar] = bars  # 原始K线
        self.__size = len(bars)
        self.freq = freq
        self.klines = {freq: self.__bars}

        if merger:
            merger = set(merger)
            merger.remove(freq)
            for m in merger:
                self.klines[m] = []
        self.klines = {freq: self.__bars}

    def __getitem__(self, index: Union[slice, int]) -> Union[List[RawBar], RawBar]:
        return self.__bars[index]

    def __len__(self):
        return self.__size

    @property
    def last(self) -> RawBar:
        return self.__bars[-1] if self.__bars else None

    def push(self, bar: RawBar):
        for freq, bars in self.klines.items():
            ts = bar.dt.timestamp()
            d = ts // freq

            dt = datetime.fromtimestamp(d * freq)
            new = RawBar(dt=dt, open=bar.open, high=bar.high, low=bar.low, close=bar.close, volume=bar.volume)

            if freq == self.freq:
                self.__size += 1

            if bars:
                last = bars[-1]
                new.index = last.index + 1
                if last.dt == dt:
                    if last.dts[-1] == bar.dt:
                        last.volume += bar.volume - last.lv
                        last.lv = bar.volume
                        if freq == self.freq:
                            self.__size -= 1
                    else:
                        last.dts.append(bar.dt)
                        last.volume += bar.volume

                    last.high = max(last.high, bar.high)
                    last.low = min(last.low, bar.low)
                    last.close = bar.close

                else:
                    bars.append(new)
                    last.done = True
                    del last.dts
                    del last.lv

            else:
                bars.append(new)

            self.calcMACD(bars)
            self.calc1355(bars)
            self.calcBOLL(bars)
            self.calcKDJ(bars)

    def calcBOLL(self, klines: List[RawBar], timeperiod: int = 20, std: int = 2):
        # https://blog.csdn.net/qq_41437512/article/details/105473845
        # https://wiki.mbalib.com/wiki/%E5%B8%83%E6%9E%97%E7%BA%BF%E6%8C%87%E6%A0%87

        self.calcMA(klines, timeperiod)
        size = len(klines)
        if size < timeperiod:
            return
        last = klines[-1]
        second = klines[-2]

        md = math.sqrt(((last.close - last.cache[f"ma_{timeperiod}"]) ** 2) / timeperiod)
        mb = second.cache[f"ma_{timeperiod}"]
        up = mb + std * md
        dn = mb - std * md
        last.cache[f"boll_{timeperiod}"] = {"up": up, "mid": mb, "dn": dn}

    def calcKDJ(self, klines: List[RawBar], timeperiod: int = 9, a=2 / 3, b=1 / 3):
        # https://wiki.mbalib.com/wiki/%E9%9A%8F%E6%9C%BA%E6%8C%87%E6%A0%87
        size = len(klines)
        if size < timeperiod:
            return
        last: RawBar = klines[-1]
        second: RawBar = klines[-2]

        l = min(klines[-timeperiod:], key=lambda x: x.low).low
        h = max(klines[-timeperiod:], key=lambda x: x.high).high
        n = h - l
        if n == 0:
            # print(self.klines[-timeperiod:])
            n = 1
            print(colored("float division by zero", "red"))
        rsv = ((last.close - l) / n) * 100
        sk = second.cache[f"K_{timeperiod}"] if second.cache.get(f"K_{timeperiod}") else 50
        sd = second.cache[f"D_{timeperiod}"] if second.cache.get(f"D_{timeperiod}") else 50
        lk = last.cache[f"K_{timeperiod}"] if last.cache.get(f"K_{timeperiod}") else 50
        ld = last.cache[f"D_{timeperiod}"] if last.cache.get(f"D_{timeperiod}") else 50
        last.cache[f"K_{timeperiod}"] = a * sk + b * rsv
        last.cache[f"D_{timeperiod}"] = a * sd + b * lk
        last.cache[f"J_{timeperiod}"] = 3 * ld - 2 * lk

    def calcEMA(self, klines, timeperiod=5):
        if len(klines) == 1:
            ema = klines[-1].close
        else:
            ema = (2 * klines[-1].close + klines[-2].cache[f"ema_{timeperiod}"] * (timeperiod - 1)) / (timeperiod + 1)
        klines[-1].cache[f"ema_{timeperiod}"] = ema

    def calcMA(self, klines, timeperiod=5):
        if len(klines) < timeperiod:
            ma = klines[-1].close
        else:
            ma = sum([k.close for k in klines[-timeperiod:]]) / timeperiod
        klines[-1].cache[f"ma_{timeperiod}"] = ma

    def calcMACD(self, klines, fastperiod=12, slowperiod=26, signalperiod=9):
        self.calcEMA(klines, fastperiod)
        self.calcEMA(klines, slowperiod)
        DIF = klines[-1].cache[f"ema_{fastperiod}"] - klines[-1].cache[f"ema_{slowperiod}"]
        klines[-1].cache[f"dif_{fastperiod}_{slowperiod}_{signalperiod}"] = DIF

        if len(klines) == 1:
            dea = klines[-1].cache[f"dif_{fastperiod}_{slowperiod}_{signalperiod}"]
        else:
            dea = (2 * klines[-1].cache[f"dif_{fastperiod}_{slowperiod}_{signalperiod}"] + klines[-2].cache[f"dea_{fastperiod}_{slowperiod}_{signalperiod}"] * (signalperiod - 1)) / (signalperiod + 1)

        klines[-1].cache[f"dea_{fastperiod}_{slowperiod}_{signalperiod}"] = dea
        klines[-1].cache[f"macd_{fastperiod}_{slowperiod}_{signalperiod}"] = (DIF - dea) * 2

    def calc1355(self, klines: List[RawBar]):
        self.calcEMA(klines, 13)
        self.calcEMA(klines, 55)
        self.calcEMA(klines, 220)
        self.calcEMA(klines, 576)
        self.calcEMA(klines, 676)


class NewBar:
    __slots__ = "dt", "open", "high", "low", "close", "volume", "index", "shape", "elements", "relation", "speck", "done", "jump", "cache", "bi", "duan", "_dt"

    def __init__(self, dt: datetime, open: float, high: float, low: float, close: float, volume: float, index: int = 0, elements=None):
        self.dt: datetime = dt
        self.open: float = open
        self.high: float = high
        self.low: float = low
        self.close: float = close
        self.volume: float = volume
        self.index: int = index
        self.shape: Optional[Shape] = None
        self.elements: Optional[List[RawBar]] = elements

        self.relation: Optional[Direction] = None  # 与前一个关系
        self.speck: Optional[float] = None  # 分型高低点
        self.done: bool = False  # 是否完成
        self.jump: bool = False  # 与前一个是否是跳空

        self.cache = dict()
        self.bi: Optional[bool] = None  # 是否是 笔
        self.duan: Optional[bool] = None  # 是否是 段
        self._dt = self.dt

    def __str__(self):
        return f"NewBar({self.dt}, {self.speck}, {self.shape})"

    def __repr__(self):
        return f"NewBar({self.dt}, {self.speck}, {self.shape})"

    @property
    def direction(self) -> Direction:
        return Direction.Up if self.open < self.close else Direction.Down

    @property
    def raw(self) -> RawBar:
        return RawBar(self.dt, self.open, self.high, self.low, self.close, self.volume)

    def candleDict(self) -> dict:
        return {
            "dt": self.dt,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "vol": self.volume,
        }

    @staticmethod
    def include(ck: "NewBar", k: RawBar, direction: Direction) -> tuple["NewBar", bool]:
        flag = False
        if double_relation(ck, k) in (Direction.Left, Direction.Right):
            if len(ck.elements) == 1:
                ck.elements[0].start_include = True  # 首次包含标志
            k.belong_include = ck.elements[0].index  # 被谁包含 索引

            if direction in (Direction.Down, Direction.JumpDown):
                # 向下取低低
                high = min(ck.high, k.high)
                low = min(ck.low, k.low)
                dt = k.dt if k.low < ck.low else ck.dt
                o = high
                c = low
            elif direction in (Direction.Up, Direction.JumpUp):
                # 向上取高高
                high = max(ck.high, k.high)
                low = max(ck.low, k.low)
                dt = k.dt if k.high > ck.high else ck.dt
                c = high
                o = low
            else:
                raise ChanException("合并方向错误")

            elements = ck.elements
            dts = [o.dt for o in elements]
            if k.dt not in dts:
                elements.append(k)
            else:
                if elements[-1].dt == k.dt:
                    elements[-1] = k
                else:
                    raise ChanException("元素重复")
            volume = sum([o.volume for o in elements])

            new = NewBar(
                dt=dt,
                open=o,
                high=high,
                low=low,
                close=c,
                index=ck.index,
                volume=volume,
                elements=elements,
            )
            flag = True
            # self.cklines[-1] = new
        else:
            new = k.new
            new.index = ck.index + 1
            ck.done = True
            # self.cklines.append(new)
        return new, flag


class FenXing:
    __slots__ = "left", "mid", "right", "index", "__shape", "__speck"

    def __init__(self, left: NewBar, mid: NewBar, right: NewBar, index: int = 0):
        self.left = left
        self.mid = mid
        self.right = right
        self.index = index

        self.__shape = mid.shape
        self.__speck = mid.speck

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


class Bi:
    __slots__ = "index", "start", "end", "elements", "done", "real_high", "real_low", "direction", "ld"

    def __init__(self, index: int, start: FenXing, end: FenXing, elements: List[NewBar], done: bool = False):
        self.index: int = index
        self.start: FenXing = start
        self.end: FenXing = end
        self.elements: List[NewBar] = elements
        self.done: bool = done

        high = self.elements[0]
        low = self.elements[0]
        for k in self.elements:
            if high.high < k.high:
                high = k
            if low.low > k.low:
                low = k
        self.real_high = high
        self.real_low = low
        if self.start.shape is Shape.G and self.end.shape is Shape.D:
            self.direction = Direction.Down
        elif self.start.shape is Shape.D and self.end.shape is Shape.G:
            self.direction = Direction.Up
        else:
            raise ChanException(self.start.shape, self.end.shape)
        self.ld: Union[dict, None] = None

    @property
    def high(self) -> float:
        return max(self.start.speck, self.end.speck)

    @property
    def low(self) -> float:
        return min(self.start.speck, self.end.speck)

    @property
    def mid(self) -> float:
        return (self.start.speck + self.end.speck) / 2

    def __str__(self):
        return f"Bi({self.direction}, {colored(self.start.dt, 'green')}, {self.start.speck}, {colored(self.end.dt, 'green')}, {self.end.speck}, {self.index})"

    def __repr__(self):
        return f"Bi({self.direction}, {colored(self.start.dt, 'green')}, {self.start.speck}, {colored(self.end.dt, 'green')}, {self.end.speck}, {self.index})"

    @property
    def relation(self) -> bool:
        if self.direction is Direction.Down:
            return double_relation(self.start, self.end) in (
                Direction.Down,
                Direction.JumpDown,
            )
        return double_relation(self.start, self.end) in (Direction.Up, Direction.JumpUp)

    @property
    def length(self) -> int:
        return len(self.elements)

    def check(self) -> bool:
        if len(self.elements) >= 5:
            assert self.start.mid is self.elements[0]
            assert self.end.mid is self.elements[-1]
            if self.direction is Direction.Down and self.start.mid is self.real_high and self.end.mid is self.real_low:
                return True
            if self.direction is Direction.Up and self.start.mid is self.real_low and self.end.mid is self.real_high:
                return True
        return False

    @staticmethod
    def append(bis, bi):
        if bis and bis[-1].end is not bi.start:
            raise TypeError("笔连续性错误")
        i = 0
        if bis:
            i = bis[-1].index + 1
        bi.index = i
        bis.append(bi)

    @staticmethod
    def pop(bis, fx):
        if bis:
            if bis[-1].end is fx:
                return bis.pop()
            else:
                raise ValueError("最后一笔终点错误", fx, bis[-1].end)

    @staticmethod
    def analyzer(fx: FenXing, fxs: List[FenXing], bis: List["Bi"], cklines: List[NewBar]):
        last = fxs[-1] if fxs else None
        left, mid, right = fx.left, fx.mid, fx.right
        if last is None:
            if mid.shape in (Shape.G, Shape.D):
                fxs.append(fx)
            return

        if last.mid.dt > fx.mid.dt:
            raise TypeError("时序错误")

        if last.shape is Shape.G and fx.shape is Shape.D:
            bi = Bi(0, last, fx, cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1])
            if bi.length > 4:
                if bi.real_high is not last.mid:
                    print("不是真顶")
                    top = bi.real_high
                    new = FenXing(cklines[cklines.index(top) - 1], top, cklines[cklines.index(top) + 1])
                    assert new.shape is Shape.G, new
                    Bi.analyzer(new, fxs, bis, cklines)  # 处理新底
                    Bi.analyzer(fx, fxs, bis, cklines)  # 再处理当前顶
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

                    if _bi:
                        nb = Bi(0, fxs[-3], _bi.start, cklines[fxs[-3].mid.index : _bi.start.mid.index + 1])
                        if not nb.check():
                            return
                        print(_bi)
                        tmp = fxs.pop()
                        assert tmp is last
                        # tmp.real = False
                        bi = Bi.pop(bis, tmp)
            else:
                ...

        elif last.shape is Shape.D and fx.shape is Shape.G:
            bi = Bi(0, last, fx, cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1])
            if bi.length > 4:
                if bi.real_low is not last.mid:
                    print("不是真底")
                    bottom = bi.real_low
                    new = FenXing(cklines[cklines.index(bottom) - 1], bottom, cklines[cklines.index(bottom) + 1])
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

                    if _bi:
                        nb = Bi(0, fxs[-3], _bi.start, cklines[fxs[-3].mid.index : _bi.start.mid.index + 1])
                        if not nb.check():
                            return
                        print(_bi)
                        tmp = fxs.pop()
                        assert tmp is last
                        # tmp.real = False
                        bi = Bi.pop(bis, tmp)
            else:
                ...

        elif last.shape is Shape.G and fx.shape is Shape.S:
            if last.speck < right.high:
                tmp = fxs.pop()
                tmp.real = False
                bi = Bi.pop(bis, tmp)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.D
                    bottom = min(cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1], key=lambda o: o.low)
                    assert bottom.shape is Shape.D
                    if last.speck > bottom.low:
                        tmp = fxs.pop()
                        tmp.real = False
                        bi = Bi.pop(bis, tmp)

                        new = FenXing(cklines[cklines.index(bottom) - 1], bottom, cklines[cklines.index(bottom) + 1])
                        assert new.shape is Shape.D, new
                        Bi.analyzer(new, fxs, bis, cklines)  # 处理新底
                        print("GS修正")

        elif last.shape is Shape.D and fx.shape is Shape.X:
            if last.speck > right.low:
                """
                底分型被突破
                1. 向上不成笔但出了高点，需要修正顶分型
                   修正后涉及循环破坏问题，即形似开口向右的扩散形态
                   解决方式
                       ①.递归调用，完全符合笔的规则，但此笔一定含有多个笔，甚至形成低级别一个走势。
                       ②.只修正一次
                2. 向上不成笔没出了高点，无需修正
                """
                tmp = fxs.pop()
                tmp.real = False
                bi = Bi.pop(bis, tmp)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.G
                    top = max(cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1], key=lambda o: o.high)
                    assert top.shape is Shape.G
                    if last.speck < top.high:
                        tmp = fxs.pop()
                        tmp.real = False
                        bi = Bi.pop(bis, tmp)

                        new = FenXing(cklines[cklines.index(top) - 1], top, cklines[cklines.index(top) + 1])
                        assert new.shape is Shape.G, new
                        Bi.analyzer(new, fxs, bis, cklines)  # 处理新顶
                        print("DX修正")

        elif last.shape is Shape.G and fx.shape is Shape.G:
            if last.speck < fx.speck:
                tmp = fxs.pop()
                tmp.real = False
                bi = Bi.pop(bis, tmp)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.D
                    bottom = min(cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1], key=lambda o: o.low)
                    assert bottom.shape is Shape.D
                    if last.speck > bottom.low:
                        tmp = fxs.pop()
                        tmp.real = False
                        bi = Bi.pop(bis, tmp)

                        new = FenXing(cklines[cklines.index(bottom) - 1], bottom, cklines[cklines.index(bottom) + 1])
                        assert new.shape is Shape.D, new
                        Bi.analyzer(new, fxs, bis, cklines)  # 处理新底
                        Bi.analyzer(fx, fxs, bis, cklines)  # 再处理当前顶
                        print("GG修正")
                        return

                if not fxs:
                    FenXing.append(fxs, fx)
                    return
                bi = Bi(0, fxs[-1], fx, cklines[cklines.index(fxs[-1].mid) : cklines.index(fx.mid) + 1])
                FenXing.append(fxs, fx)
                Bi.append(bis, bi)

        elif last.shape is Shape.D and fx.shape is Shape.D:
            if last.speck > fx.speck:
                tmp = fxs.pop()
                tmp.real = False
                bi = Bi.pop(bis, tmp)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.G
                    top = max(cklines[cklines.index(last.mid) : cklines.index(fx.mid) + 1], key=lambda o: o.high)
                    assert top.shape is Shape.G
                    if last.speck < top.high:
                        tmp = fxs.pop()
                        tmp.real = False
                        bi = Bi.pop(bis, tmp)

                        new = FenXing(cklines[cklines.index(top) - 1], top, cklines[cklines.index(top) + 1])
                        assert new.shape is Shape.G, new
                        Bi.analyzer(new, fxs, bis, cklines)  # 处理新顶
                        Bi.analyzer(fx, fxs, bis, cklines)  # 再处理当前底
                        print("DD修正")
                        return

                if not fxs:
                    FenXing.append(fxs, fx)
                    return
                bi = Bi(0, fxs[-1], fx, cklines[cklines.index(fxs[-1].mid) : cklines.index(fx.mid) + 1])
                FenXing.append(fxs, fx)
                Bi.append(bis, bi)

        elif last.shape is Shape.G and fx.shape is Shape.X:
            ...

        elif last.shape is Shape.D and fx.shape is Shape.S:
            ...

        else:
            raise ValueError(last.shape, fx.shape)

    @staticmethod
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



class Duan:
    __slots__ = "index", "__start", "__end", "elements", "done", "pre", "features", "info", "direction", "level"

    def __init__(self, index: int, start: FenXing, end: FenXing, elements: List[Bi]):
        self.index: int = index
        self.__start: FenXing = start
        self.__end: FenXing = end
        self.elements: List[Bi] = elements

        self.done: bool = False
        self.pre: Optional[Self] = None
        if self.__start.shape is Shape.G and self.__end.shape is Shape.D:
            self.direction = Direction.Down
        elif self.__start.shape is Shape.D and self.__end.shape is Shape.G:
            self.direction = Direction.Up
        else:
            raise ChanException(self.start, self.end)

        self.features: list[Optional[FeatureSequence]] = [None, None, None]
        self.info = []

        self.level = 1
        if type(self.elements[0]) is self:
            self.level = 2

    def __str__(self):
        return f"Duan({self.index}, {self.direction}, {len(self.elements)}, 完成否:{self.done}, {self.pre is not None}, {self.start}, {self.end})"

    def __repr__(self):
        return f"Duan({self.index}, {self.direction}, {len(self.elements)}, 完成否:{self.done}, {self.pre is not None}, {self.start}, {self.end})"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        return value

    @property
    def end(self) -> FenXing:
        return self.__end

    @property
    def start(self) -> FenXing:
        return self.__start

    @start.setter
    def start(self, value: FenXing) -> None:
        self.__start = value

    @end.setter
    def end(self, value: FenXing) -> None:
        if self.__start.shape is value.shape:
            raise ChanException("分型相同")
        self.__end = value

    @property
    def lmr(self) -> tuple[bool, bool, bool]:
        return self.left is not None, self.mid is not None, self.right is not None

    @property
    def state(self) -> States:
        if self.pre is not None:
            return "老阳" if self.direction is Direction.Down else "老阴"
        else:
            return "少阳" if self.direction is Direction.Up else "少阴"

    @property
    def left(self) -> "FeatureSequence":
        return self.features[0]

    @property
    def mid(self) -> "FeatureSequence":
        return self.features[1]

    @property
    def right(self) -> "FeatureSequence":
        return self.features[2]

    @property
    def high(self) -> float:
        return max(self.__start.speck, self.__end.speck)

    @property
    def low(self) -> float:
        return min(self.__start.speck, self.__end.speck)

    @classmethod
    def new(cls, index, obj):
        return cls(
            index,
            obj.start,
            obj.end,
            [
                obj,
            ],
        )

    def charts(self):
        return [
            {"xd": self.__start.speck, "dt": self.__start.dt},
            {"xd": self.__end.speck, "dt": self.__end.dt},
        ]

    def get_elements(self) -> Iterable[Bi]:
        elements = []
        for obj in self.elements:
            elements.append(obj)
            if obj.end is self.end:
                break
        return elements

    def append_element(self, bi: Bi):
        if self.elements[-1].end is bi.start:
            self.elements.append(bi)
        else:
            dp("线段添加元素时，元素不连续", self.elements[-1], bi)
            raise ChanException("线段添加元素时，元素不连续", self.elements[-1], bi)

    def pop_element(self, bi: Bi) -> bool:
        if self.elements[-1] is bi:
            self.elements.pop()
            return True
        else:
            raise ChanException("线段弹出元素时，元素不匹配")

    def set_done(self, fx: FenXing):
        elements = []
        for obj in self.elements:
            if elements:
                elements.append(obj)
            if obj.start is fx:
                elements.append(obj)
        self.end = fx
        self.done = True
        return elements

    def check(self):
        if not ((self.__start.shape is Shape.G and self.__end.shape is Shape.D) or (self.__start.shape is Shape.D and self.__end.shape is Shape.G)):
            raise ChanException
        if len(self.elements) >= 3:
            if double_relation(self.elements[0], self.elements[2]) in (Direction.JumpUp, Direction.JumpDown):
                raise ChanException("线段前3笔没有重叠")

    @staticmethod
    def analyzer(bi: Bi, xds: list["Duan"], level: int = 0):
        cmd = "Duans.PUSH"
        new = Duan(0, bi.start, bi.end, [bi])
        if not xds:
            xds.append(new)
            return
        duan: Duan = xds[-1]
        state: States = duan.state
        last: Optional[Duan] = duan.pre
        # last = duans[-2] if len(duans) > 1 else last
        left: Optional[FeatureSequence] = duan.features[0]
        mid: Optional[FeatureSequence] = duan.features[1]
        # right: Optional[FeatureSequence] = duan.features[2]
        lmr: Tuple[bool, bool, bool] = duan.lmr

        ddp("    " * level, cmd, state, lmr, duan, bi)
        ddp("    " * level, duan.features)
        ddp("    " * level, duan.elements)

        duan.append_element(bi)
        if duan.direction is bi.direction:
            if duan.mid:
                duan.end = duan.mid.start
            else:
                duan.end = bi.end
            ddp("    " * level, "方向相同, 更新结束点", duan.end, duan.state)
            return

        feature = FeatureSequence({bi}, Direction.Up if bi.direction is Direction.Down else Direction.Down)
        if lmr == (False, False, False):
            assert feature.direction is duan.direction
            duan.features = [feature, None, None]

        elif lmr == (True, False, False):
            assert left.direction is duan.direction
            relation = double_relation(left, bi)
            ddp("    " * level, "第二特征序列", relation, duan.state)
            if relation is Direction.Left:
                left.add(bi)
            elif relation is Direction.Right:
                if last is not None:
                    left.add(bi)
                else:
                    duan.features = [left, feature, None]

            elif relation in (Direction.Up, Direction.JumpUp):
                if duan.direction is Direction.Up:
                    duan.features = [left, feature, None]
                else:
                    # Down
                    duan.features = [left, feature, None]

            elif relation in (Direction.Down, Direction.JumpDown):
                if duan.direction is Direction.Down:
                    duan.features = [left, feature, None]
                else:
                    # Up
                    duan.features = [left, feature, None]

            else:
                raise ChanException("未知的关系", relation)

        elif lmr == (True, True, False):
            assert mid.direction is duan.direction
            relation = double_relation(mid, bi)
            ddp("    " * level, "第三特征序列", relation, duan.state)
            if relation is Direction.Left:
                mid.add(bi)

            elif relation is Direction.Right:
                if last is not None:
                    mid.add(bi)
                else:
                    duan.features = [mid, feature, None]

            elif relation in (Direction.Up, Direction.JumpUp):
                if duan.direction is Direction.Up:
                    duan.features = [mid, feature, None]
                else:
                    # Down, 底分型
                    duan.features = [left, mid, feature]
                    duan.end = mid.start
                    duan.done = True
                    elements = duan.set_done(mid.start)

                    new = Duan.new(duan.index + 1, elements[0])
                    new.elements = elements
                    new.end = elements[-1].end
                    if double_relation(left, mid) is Direction.JumpDown:
                        duan.pre = new
                    ddp("    " * level, "底分型终结", duan.pre is not None, elements[0].direction)
                    features = FeatureSequence.analysis(elements, new.direction)
                    if features:
                        new.features = [features[-1], None, None]
                        if len(features) > 1:
                            new.features = [features[-2], features[-1], None]
                    if duan.end is new.start:
                        xds.append(new)

                    else:
                        raise ChanException("线段不连续", duan.elements[-1].end, new.elements[0].start)

            elif relation in (Direction.Down, Direction.JumpDown):
                if duan.direction is Direction.Down:
                    duan.features = [mid, feature, None]
                else:
                    # Up, 顶分型
                    duan.features = [left, mid, feature]
                    duan.end = mid.start
                    duan.done = True
                    elements = duan.set_done(mid.start)

                    new = Duan.new(duan.index + 1, elements[0])
                    new.elements = elements
                    new.end = elements[-1].end
                    if double_relation(left, mid) is Direction.JumpUp:
                        duan.pre = new
                    ddp("    " * level, "顶分型终结", duan.pre is not None, elements[0].direction)
                    features = FeatureSequence.analysis(elements, new.direction)
                    if features:
                        new.features = [features[-1], None, None]
                        if len(features) > 1:
                            new.features = [features[-2], features[-1], None]

                    if duan.end is new.start:
                        xds.append(new)
                    else:
                        raise ChanException("线段不连续", duan.elements[-1].end, new.elements[0].start)

            else:
                raise ChanException("未知的关系", relation)

        else:
            raise ChanException("未知的状态", state, lmr)

        # duans[-1].check()


class ZhongShu:
    # __slots__ = "elements", "index", "level"

    def __init__(self, obj: Union[Bi, Duan]):
        self.elements = [obj]
        self.index = 0
        self.level = 0
        self._doing = None
        if not obj.done:
            self._doing = obj
        if type(obj) is Bi:
            self.level = 1
        if type(obj) is Duan:
            self.level = 2

    def __str__(self):
        return f"中枢({self.elements})"

    def __repr__(self):
        return f"中枢({self.elements})"

    @property
    def left(self) -> Union[Bi, Duan]:
        return self.elements[0] if self.elements else None

    @property
    def mid(self) -> Union[Bi, Duan]:
        return self.elements[1] if len(self.elements) > 1 else None

    @property
    def right(self) -> Union[Bi, Duan]:
        return self.elements[2] if len(self.elements) > 2 else None

    @property
    def last(self) -> Union[Bi, Duan]:
        return self.elements[-1] if self.elements else None

    @property
    def direction(self) -> Direction:
        return Direction.Down if self.start.shape is Shape.D else Direction.Up

    @property
    def zg(self) -> float:
        return min(self.elements[:3], key=lambda o: o.high).high

    @property
    def zd(self) -> float:
        return max(self.elements[:3], key=lambda o: o.low).low

    @property
    def g(self) -> float:
        return min(self.elements, key=lambda o: o.high).high

    @property
    def d(self) -> float:
        return max(self.elements, key=lambda o: o.low).low

    @property
    def gg(self) -> float:
        return max(self.elements, key=lambda o: o.high).high

    @property
    def dd(self) -> float:
        return min(self.elements, key=lambda o: o.low).low

    def check(self) -> bool:
        return double_relation(self.left, self.right) in (
            Direction.Down,
            Direction.Up,
            Direction.Left,
            Direction.Right,
        )

    @property
    def high(self) -> float:
        return self.zg

    @property
    def low(self) -> float:
        return self.zd

    @property
    def start(self) -> FenXing:
        return self.left.start

    @property
    def end(self) -> FenXing:
        return self.elements[-1].end

    def pop_element(self, obj: Union[Bi, Duan]):
        if self.last.start is obj.start:
            if self.last is not obj:
                dp("警告：中枢元素不匹配!!!", self.last, obj)
            self.elements.pop()
        else:
            raise ChanException("中枢无法删除元素", self.last, obj)

    def append_element(self, obj: Union[Bi, Duan]):
        # dp("添加中枢元素", obj)
        # dp("现有中枢元素", self.elements)
        if self.last.end is obj.start:
            self.elements.append(obj)
            if not obj.done:
                self._doing = obj
        else:
            raise ChanException("中枢无法添加元素", self.last, obj)

    @staticmethod
    def analyzer(elements: List[Union[Bi, Duan]]) -> tuple[bool, list]:
        if len(elements) < 3:
            return False, []
        direction = elements[0].direction
        flag = False
        zss: List[Union[Bi, Duan, ZhongShu]] = [elements[0]]
        for obj in elements[1:]:
            last: Union[Bi, Duan, ZhongShu] = zss[-1]
            zs = [o for o in zss if type(o) is ZhongShu]
            last_zs: ZhongShu = zs[-1] if zs else None
            if last_zs and last_zs.elements[-1].end is obj.start:
                if double_relation(last_zs, obj) in (Direction.JumpUp, Direction.JumpDown):
                    if last_zs.right is not None:
                        new_zs = ZhongShu(obj)
                        zss.append(new_zs)
                    else:
                        o = last_zs.elements.pop(0)
                        zss.insert(zss.index(last_zs), o)
                else:
                    last_zs.append_element(obj)
            else:
                new_zs = ZhongShu(obj)
                zss.append(new_zs)
        return True, zss

    def charts(self):
        return [
            [
                self.start.mid.dt,
                self.start.mid.dt,
                self.elements[-1].start.mid.dt,
                self.elements[-1].start.mid.dt,
                self.start.mid.dt,
            ]
            if len(self.elements) > 3
            else [
                self.start.mid.dt,
                self.start.mid.dt,
                self.end.mid.dt,
                self.end.mid.dt,
                self.start.mid.dt,
            ],
            [self.zg, self.zd, self.zd, self.zg, self.zg],
            "#993333" if self.direction is Direction.Up else "#99CC99",  # 上下上 为 红色，反之为 绿色
            self.level,
        ]

    def charts_jhl(self):
        return [
            [
                self.start.mid.dt,
                self.start.mid.dt,
                self.end.mid.dt,
                self.end.mid.dt,
                self.start.mid.dt,
            ],
            [self.zg, self.zd, self.zd, self.zg, self.zg],
            "#CC0033" if self.direction is Direction.Up else "#66CC99",
            self.level + 2,
        ]


@dataclass
class ZouShi:
    body: List[Union[Duan, Bi, ZhongShu]]


class FeatureSequence:
    def __init__(self, elements: set, direction: Direction):
        self.__elements: set = elements
        self.direction: Direction = direction  # 线段方向
        self.shape: Optional[Shape] = None
        self.index = 0

    def __str__(self):
        if not self.__elements:
            return f"空特征序列({self.direction})"
        return f"特征序列({self.direction}, {self.start.dt}, {self.end.dt}, {len(self.__elements)})"

    def __repr__(self):
        if not self.__elements:
            return f"空特征序列({self.direction})"
        return f"特征序列({self.direction}, {self.start.dt}, {self.end.dt}, {len(self.__elements)})"

    def __len__(self):
        return len(self.__elements)

    def __iter__(self):
        return iter(self.__elements)

    def add(self, obj: Union[Bi, Duan]):
        direction = Direction.Down if self.direction is Direction.Up else Direction.Up
        if obj.direction is not direction:
            raise ChanException("方向不匹配", direction, obj, self)
        self.__elements.add(obj)

    def remove(self, obj: Union[Bi, Duan]):
        direction = Direction.Down if self.direction is Direction.Up else Direction.Up
        if obj.direction is not direction:
            raise ChanException("方向不匹配", direction, obj, self)
        self.__elements.remove(obj)

    @property
    def start(self) -> FenXing:
        if not self.__elements:
            raise ChanException("数据异常", self)
        func = min
        if self.direction is Direction.Up:  # 线段方向向上特征序列取高高
            func = max
        if self.direction is Direction.Down:
            func = min
        fx = func([obj.start for obj in self.__elements], key=lambda fx: fx.speck)
        assert fx.shape in (Shape.G, Shape.D)
        return fx

    @property
    def end(self) -> FenXing:
        if not self.__elements:
            raise ChanException("数据异常", self)
        func = min
        if self.direction is Direction.Up:  # 线段方向向上特征序列取高高
            func = max
        if self.direction is Direction.Down:
            func = min
        fx = func([obj.end for obj in self.__elements], key=lambda fx: fx.speck)
        assert fx.shape in (Shape.G, Shape.D)
        return fx

    @property
    def high(self) -> float:
        return max([self.end, self.start], key=lambda fx: fx.speck).speck

    @property
    def low(self) -> float:
        return min([self.end, self.start], key=lambda fx: fx.speck).speck

    @staticmethod
    def analysis(bis: list, direction: Direction):
        result: List[FeatureSequence] = []
        for obj in bis:
            if obj.direction is direction:
                continue
            if result:
                last = result[-1]

                if double_relation(last, obj) in (Direction.Left,):
                    last.add(obj)
                else:
                    result.append(FeatureSequence({obj}, Direction.Up if obj.direction is Direction.Down else Direction.Down))
                    # dp("FS.ANALYSIS", double_relation(last, obj))
            else:
                result.append(FeatureSequence({obj}, Direction.Up if obj.direction is Direction.Down else Direction.Down))
        return result


class KlineGenerator:
    def __init__(self, arr=[3, 2, 5, 3, 7, 4, 7, 2.5, 5, 4, 8, 6]):
        self.dt = datetime(2021, 9, 3, 19, 50, 40, 916152)
        self.arr = arr

    def up(self, start, end, size=5):
        n = 0
        m = round(abs(start - end) * (1 / size), 8)
        o = start
        # c = round(o + m, 4)

        while n < size:
            c = round(o + m, 4)
            yield RawBar(self.dt, o, c, o, c, 1)
            o = c
            n += 1
            self.dt = datetime.fromtimestamp(self.dt.timestamp() + 60 * 60)

    def down(self, start, end, size=5):
        n = 0
        m = round(abs(start - end) * (1 / size), 8)
        o = start
        # c = round(o - m, 4)

        while n < size:
            c = round(o - m, 4)
            yield RawBar(self.dt, o, o, c, c, 1)
            o = c
            n += 1
            self.dt = datetime.fromtimestamp(self.dt.timestamp() + 60 * 60)

    @property
    def result(self):
        size = len(self.arr)
        i = 0
        # sizes = [5 for i in range(l)]
        result = []
        while i + 1 < size:
            s = self.arr[i]
            e = self.arr[i + 1]
            if s > e:
                for k in self.down(s, e):
                    result.append(k)
            else:
                for k in self.up(s, e):
                    result.append(k)
            i += 1
        return result


class BaseAnalyzer:
    def __init__(self, symbol: str, freq: int):
        self.__symbol = symbol
        self.__freq = freq
        self._raws: List[RawBar] = []
        self._news: List[NewBar] = []
        self._fxs: List[FenXing] = []
        self._bis: List[Bi] = []
        self._bi_zss: List[ZhongShu] = []
        self._duans: List[Duan] = []
        self._duan_zss: List[ZhongShu] = []

        self._zss: List[ZouShi] = []  # 走势

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def freq(self) -> int:
        return self.__freq

    def push(self, bar: RawBar, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        last = self._news[-1] if self._news else None
        news = self._news
        if last is None:
            news.append(bar.new)
        else:
            relation = double_relation(last, bar)
            if relation in (Direction.Left, Direction.Right):
                direction = last.direction
                try:
                    direction = double_relation(news[-2], last)
                except IndexError:
                    traceback.print_exc()

                new, flag = NewBar.include(last, bar, direction)
                new.index = last.index
                news[-1] = new
            else:
                new = bar.new
                new.index = last.index + 1
                news.append(new)
                
        klines = news
        if len(klines) == 1:
            ema_slow = klines[-1].close
            ema_fast = klines[-1].close
        else:
            ema_slow = (2 * klines[-1].close + klines[-2].cache[f"ema_{slow_period}"] * (slow_period - 1)) / (slow_period + 1)
            ema_fast = (2 * klines[-1].close + klines[-2].cache[f"ema_{fast_period}"] * (fast_period - 1)) / (fast_period + 1)
        klines[-1].cache[f"ema_{slow_period}"] = ema_slow
        klines[-1].cache[f"ema_{fast_period}"] = ema_fast
        DIF = ema_fast - ema_slow
        klines[-1].cache[f"dif_{fast_period}_{slow_period}_{signal_period}"] = DIF

        if len(klines) == 1:
            dea = DIF
        else:
            dea = (2 * DIF + klines[-2].cache[f"dea_{fast_period}_{slow_period}_{signal_period}"] * (signal_period - 1)) / (signal_period + 1)

        klines[-1].cache[f"dea_{fast_period}_{slow_period}_{signal_period}"] = dea
        macd = (DIF - dea) * 2
        klines[-1].cache[f"macd_{fast_period}_{slow_period}_{signal_period}"] = macd
        
        try:
            left, mid, right = news[-3:]
        except ValueError:
            return

        left, mid, right = news[-3:]  # ValueError: not enough values to unpack (expected 3, got 2)
        shape, relations = triple_relation(left, mid, right)
        mid.shape = shape
        if relations[1] in (Direction.JumpDown, Direction.JumpUp):
            right.jump = True
        if relations[0] in (Direction.JumpDown, Direction.JumpUp):
            mid.jump = True

        if shape is Shape.G:
            mid.speck = mid.high
            fx = FenXing(left, mid, right)
            self.__analysis_fx(fx, self._news, 0)
        if shape is Shape.D:
            mid.speck = mid.low
            fx = FenXing(left, mid, right)
            self.__analysis_fx(fx, self._news, 0)

    def __pop_bi(self, fx, level: int):
        cmd = "Bis.POP"
        bdp("    " * level, cmd, fx)
        last = self._bis[-1] if self._bis else None
        if last:
            if last.end is fx:
                bi = self._bis.pop()
                bdp("    " * level, cmd, bi)
                fx.mid.bi = False
                self.__pop_bi_zs(bi)
                self.__pop_duan(bi, level)

            else:
                raise ChanException("最后一笔终点错误", fx, last.end)
        else:
            bdp("    " * level, cmd, "空")

    def __push_bi(self, bi: Bi, level: int):
        cmd = "Bis.PUSH"
        bdp("    " * level, cmd, bi)
        last = self._bis[-1] if self._bis else None

        if last and last.end is not bi.start:
            raise ChanException("笔连续性错误")
        i = 0
        if last:
            i = last.index + 1
        bi.index = i
        self._bis.append(bi)
        bi.start.mid.bi = True
        bi.end.mid.bi = True
        self.__push_bi_zs(bi)
        self.__push_duan(bi, level)

    def __analysis_fx(self, fx: FenXing, cklines: List[NewBar], level: int):
        cmd = "Bis.ANALYSIS"
        bdp("    " * level, cmd, fx)
        fxs: List[FenXing] = self._fxs

        last: Union[FenXing, None] = fxs[-1] if fxs else None
        _, mid, right = fx.left, fx.mid, fx.right
        if last is None:
            bdp("    " * level, cmd, "首次分析")
            if mid.shape in (Shape.G, Shape.D):
                fxs.append(fx)
            return

        if last.mid.dt > fx.mid.dt:
            raise ChanException("时序错误")

        if last.shape is Shape.G and fx.shape is Shape.D:
            bdp("    " * level, cmd, "GD")
            bi = Bi(0, last, fx, cklines[last.mid.index : fx.mid.index + 1])
            if bi.length > 4:
                if bi.real_high is not last.mid:
                    dp("    " * level, cmd, "不是真顶")
                    top = bi.real_high
                    new = FenXing(cklines[top.index - 1], top, cklines[top.index + 1])
                    assert new.shape is Shape.G, new
                    self.__analysis_fx(new, cklines, level + 1)  # 处理新底
                    self.__analysis_fx(fx, cklines, level + 1)  # 再处理当前顶
                    return
                flag = bi.relation
                if flag and fx.mid is bi.real_low:
                    FenXing.append(fxs, fx)
                    self.__push_bi(bi, level)
                else:
                    ...
                    # 2024 05 21 修正
                    _cklines = cklines[last.mid.index :]
                    _fx, _bi = Bi.analysis_one(_cklines)

                    if _bi:
                        nb = Bi(0, fxs[-3], _bi.start, cklines[fxs[-3].mid.index : _bi.start.mid.index + 1])
                        if not nb.check():
                            return
                        print(_bi)
                        tmp = fxs.pop()
                        assert tmp is last
                        # tmp.real = False
                        self.__pop_bi(tmp, level)
                        self.__analysis_fx(_bi.start, cklines, level + 1)  # 处理新底
                        self.__analysis_fx(_bi.end, cklines, level + 1)  # 再处理当前顶
            else:
                ...

        elif last.shape is Shape.D and fx.shape is Shape.G:
            bdp("    " * level, cmd, "DG")
            bi = Bi(0, last, fx, cklines[last.mid.index : fx.mid.index + 1])
            if bi.length > 4:
                if bi.real_low is not last.mid:
                    dp("    " * level, cmd, "不是真底")
                    bottom = bi.real_low
                    new = FenXing(cklines[bottom.index - 1], bottom, cklines[bottom.index + 1])
                    assert new.shape is Shape.D, new
                    self.__analysis_fx(new, cklines, level + 1)  # 处理新底
                    self.__analysis_fx(fx, cklines, level + 1)  # 再处理当前顶
                    return
                flag = bi.relation
                if flag and fx.mid is bi.real_high:
                    FenXing.append(fxs, fx)
                    self.__push_bi(bi, level)
                else:
                    ...
                    # 2024 05 21 修正
                    _cklines = cklines[last.mid.index :]
                    _fx, _bi = Bi.analysis_one(_cklines)

                    if _bi:
                        nb = Bi(0, fxs[-3], _bi.start, cklines[fxs[-3].mid.index : _bi.start.mid.index + 1])
                        if not nb.check():
                            return
                        print(_bi)
                        tmp = fxs.pop()
                        assert tmp is last
                        # tmp.real = False
                        self.__pop_bi(tmp, level)
                        self.__analysis_fx(_bi.start, cklines, level + 1)  # 处理新底
                        self.__analysis_fx(_bi.end, cklines, level + 1)  # 再处理当前顶
            else:
                ...

        elif last.shape is Shape.G and fx.shape is Shape.S:
            if last.speck < right.high:
                tmp = fxs.pop()
                assert tmp is last
                # tmp.real = False
                self.__pop_bi(tmp, level)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.D
                    bottom = min(cklines[last.mid.index : fx.mid.index + 1], key=lambda o: o.low)
                    assert bottom.shape is Shape.D
                    if last.speck > bottom.low:
                        tmp = fxs.pop()
                        assert tmp is last
                        # tmp.real = False
                        self.__pop_bi(tmp, level)

                        new = FenXing(cklines[bottom.index - 1], bottom, cklines[bottom.index + 1])
                        assert new.shape is Shape.D, new
                        dp("    " * level, cmd, "GS修正")
                        self.__analysis_fx(new, cklines, level + 1)  # 处理新底

        elif last.shape is Shape.D and fx.shape is Shape.X:
            if last.speck > right.low:
                """
                底分型被突破
                1. 向上不成笔但出了高点，需要修正顶分型
                   修正后涉及循环破坏问题，即形似开口向右的扩散形态
                   解决方式
                       ①.递归调用，完全符合笔的规则，但此笔一定含有多个笔，甚至形成低级别一个走势。
                       ②.只修正一次
                2. 向上不成笔没出了高点，无需修正
                """
                tmp = fxs.pop()
                assert tmp is last
                # tmp.real = False
                self.__pop_bi(tmp, level)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.G
                    top = max(cklines[last.mid.index : fx.mid.index + 1], key=lambda o: o.high)
                    assert top.shape is Shape.G
                    if last.speck < top.high:
                        tmp = fxs.pop()
                        assert tmp is last
                        # tmp.real = False
                        self.__pop_bi(tmp, level)
                        new = FenXing(cklines[top.index - 1], top, cklines[top.index + 1])
                        assert new.shape is Shape.G, new
                        dp("    " * level, cmd, "DX修正")
                        self.__analysis_fx(new, cklines, level + 1)  # 处理新顶

        elif last.shape is Shape.G and fx.shape is Shape.G:
            if last.speck < fx.speck:
                tmp = fxs.pop()
                assert tmp is last
                # tmp.real = False
                self.__pop_bi(tmp, level)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.D
                    bottom = min(cklines[last.mid.index : fx.mid.index + 1], key=lambda o: o.low)
                    assert bottom.shape is Shape.D
                    if last.speck > bottom.low:
                        tmp = fxs.pop()
                        assert tmp is last
                        # tmp.real = False
                        self.__pop_bi(tmp, level)
                        new = FenXing(cklines[bottom.index - 1], bottom, cklines[bottom.index + 1])
                        assert new.shape is Shape.D, new
                        dp("    " * level, cmd, "GG修正")
                        self.__analysis_fx(new, cklines, level + 1)  # 处理新底
                        self.__analysis_fx(fx, cklines, level + 1)  # 再处理当前顶
                        return

                if not fxs:
                    FenXing.append(fxs, fx)
                    return
                bi = Bi(0, fxs[-1], fx, cklines[fxs[-1].mid.index : fx.mid.index + 1])
                FenXing.append(fxs, fx)
                self.__push_bi(bi, level)

        elif last.shape is Shape.D and fx.shape is Shape.D:
            if last.speck > fx.speck:
                tmp = fxs.pop()
                assert tmp is last
                # tmp.real = False
                self.__pop_bi(tmp, level)

                if fxs:
                    # 查找
                    last = fxs[-1]
                    assert last.shape is Shape.G
                    top = max(cklines[last.mid.index : fx.mid.index + 1], key=lambda o: o.high)
                    assert top.shape is Shape.G
                    if last.speck < top.high:
                        tmp = fxs.pop()
                        assert tmp is last
                        # tmp.real = False
                        self.__pop_bi(tmp, level)
                        new = FenXing(cklines[top.index - 1], top, cklines[top.index + 1])
                        assert new.shape is Shape.G, new
                        dp("    " * level, cmd, "DD修正")
                        self.__analysis_fx(new, cklines, level + 1)  # 处理新顶
                        self.__analysis_fx(fx, cklines, level + 1)  # 再处理当前底
                        return

                if not fxs:
                    FenXing.append(fxs, fx)
                    return
                bi = Bi(0, fxs[-1], fx, cklines[fxs[-1].mid.index : fx.mid.index + 1])
                FenXing.append(fxs, fx)
                self.__push_bi(bi, level)

        elif last.shape is Shape.G and fx.shape is Shape.X:
            ...

        elif last.shape is Shape.D and fx.shape is Shape.S:
            ...

        else:
            raise ChanException(last.shape, fx.shape)

    def __pop_duan(self, bi, level=0):
        ddp()

        duans: List[Duan] = self._duans
        cmd = "Duans.POP"

        duan: Duan = duans[-1]
        state: States = duan.state
        last: Optional[Duan] = duan.pre
        last = duans[-2] if len(duans) > 1 else last
        left: Optional[FeatureSequence] = duan.features[0]
        mid: Optional[FeatureSequence] = duan.features[1]
        right: Optional[FeatureSequence] = duan.features[2]
        lmr: Tuple[bool, bool, bool] = duan.lmr

        ddp("    " * level, cmd, state, lmr, duan, bi)
        ddp("    " * level, duan.features)
        ddp("    " * level, duan.elements)

        duan.pop_element(bi)

        if last is not None:
            if (last.right and bi in last.right) or (last.right is None and bi in last.left):
                # Duan.pop(duans, duan, ZShandler)
                assert duans.pop() is duan
                self.__pop_duan_zs(duan)
                last.pop_element(bi)
                last.features = [last.left, last.mid, None]
                return

        if lmr == (False, False, False):
            if len(duan.elements) >= 1:
                raise ChanException("线段中有多个元素，但特征序列为空")
            # Duan.pop(duans, duan, ZShandler)
            assert duans.pop() is duan
            self.__pop_duan_zs(duan)

            if last is not None:
                last.pop_element(bi)

        elif lmr == (True, False, False):
            if duan.direction is bi.direction:
                return

            left.remove(bi)
            if not left:
                duan.features = [None, None, None]

        elif lmr == (True, True, False):
            if duan.direction is bi.direction:
                return
            features = FeatureSequence.analysis(duan.elements, duan.direction)
            mid.remove(bi)
            if not mid:
                duan.features = [left, None, None]
            else:
                duan.features = [left, mid, None]
            if len(features) >= 2:
                if left in features:
                    ddp("    " * level, cmd, state, "第二特征序列 修正", features)
                    duan.features = [features[-2], features[-1], None]

        elif lmr == (True, True, True):
            if duan.direction is bi.direction:
                return
            right.remove(bi)
            if right:
                raise ChanException("右侧特征序列不为空")
            duan.features = [left, mid, None]

        else:
            raise ChanException("未知的状态", state, lmr)

    def __push_duan(self, bi: Bi, level=0):
        ddp()

        duans: List[Duan] = self._duans
        cmd = "Duans.PUSH"
        if not duans:
            duan = Duan.new(0, bi)
            duans.append(duan)
            self.__push_duan_zs(duan)
            return

        duan: Duan = duans[-1]
        state: States = duan.state
        last: Optional[Duan] = duan.pre
        # last = duans[-2] if len(duans) > 1 else last
        left: Optional[FeatureSequence] = duan.features[0]
        mid: Optional[FeatureSequence] = duan.features[1]
        # right: Optional[FeatureSequence] = duan.features[2]
        lmr: Tuple[bool, bool, bool] = duan.lmr

        ddp("    " * level, cmd, state, lmr, duan, bi)
        ddp("    " * level, duan.features)
        ddp("    " * level, duan.elements)

        duan.append_element(bi)
        if duan.direction is bi.direction:
            ddp("    " * level, "方向相同, 更新结束点")
            if duan.mid:
                duan.end = duan.mid.start
            else:
                duan.end = bi.end
            dp("    " * level, "方向相同, 更新结束点", duan.end)
            return

        feature = FeatureSequence({bi}, Direction.Up if bi.direction is Direction.Down else Direction.Down)
        if lmr == (False, False, False):
            assert feature.direction is duan.direction
            duan.features = [feature, None, None]

        elif lmr == (True, False, False):
            assert left.direction is duan.direction
            relation = double_relation(left, bi)
            ddp("    " * level, "第二特征序列", relation)
            if relation is Direction.Left:
                left.add(bi)
            elif relation is Direction.Right:
                if last is not None:
                    left.add(bi)
                else:
                    duan.features = [left, feature, None]

            elif relation in (Direction.Up, Direction.JumpUp):
                if duan.direction is Direction.Up:
                    duan.features = [left, feature, None]
                else:
                    # Down
                    duan.features = [left, feature, None]
                    return
                    duan.end = bi.start
                    new = Duan.new(duan.index + 1, bi)
                    # Duan.append(duans, new, ZShandler)
                    if duan.end is new.start:
                        duans.append(new)
                        self.__push_duan_zs(new)
                    else:
                        raise ChanException("线段不连续", duan.elements[-1].end, new.elements[0].start)

            elif relation in (Direction.Down, Direction.JumpDown):
                if duan.direction is Direction.Down:
                    duan.features = [left, feature, None]
                else:
                    # Up
                    duan.features = [left, feature, None]
                    return
                    duan.end = bi.start
                    new = Duan.new(duan.index + 1, bi)
                    # Duan.append(duans, new, ZShandler)
                    if duan.end is new.start:
                        duans.append(new)
                        self.__push_duan_zs(new)
                    else:
                        raise ChanException("线段不连续", duan.elements[-1].end, new.elements[0].start)

            else:
                raise ChanException("未知的关系", relation)

        elif lmr == (True, True, False):
            assert mid.direction is duan.direction
            relation = double_relation(mid, bi)
            ddp("    " * level, "第三特征序列", relation)
            if relation is Direction.Left:
                mid.add(bi)

            elif relation is Direction.Right:
                if last is not None:
                    mid.add(bi)
                else:
                    duan.features = [mid, feature, None]

            elif relation in (Direction.Up, Direction.JumpUp):
                if duan.direction is Direction.Up:
                    duan.features = [mid, feature, None]
                else:
                    # Down, 底分型
                    duan.features = [left, mid, feature]
                    duan.end = mid.start
                    duan.done = True
                    elements = duan.set_done(mid.start)
                    if self._duan_zss:
                        self.__pop_duan_zs(duan)
                        self.__push_duan_zs(duan)
                    new = Duan.new(duan.index + 1, elements[0])
                    new.elements = elements
                    new.end = elements[-1].end
                    if double_relation(left, mid) is Direction.JumpDown:
                        duan.pre = new
                    ddp("    " * level, "底分型终结", duan.pre is not None, elements[0].direction)
                    features = FeatureSequence.analysis(elements, new.direction)
                    if features:
                        new.features = [features[-1], None, None]
                        if len(features) > 1:
                            new.features = [features[-2], features[-1], None]
                    # Duan.append(duans, new, ZShandler)
                    if duan.end is new.start:
                        duans.append(new)
                        self.__push_duan_zs(new)
                    else:
                        raise ChanException("线段不连续", duan.elements[-1].end, new.elements[0].start)

            elif relation in (Direction.Down, Direction.JumpDown):
                if duan.direction is Direction.Down:
                    duan.features = [mid, feature, None]
                else:
                    # Up, 顶分型
                    duan.features = [left, mid, feature]
                    duan.end = mid.start
                    duan.done = True
                    elements = duan.set_done(mid.start)
                    if self._duan_zss:
                        self.__pop_duan_zs(duan)
                        self.__push_duan_zs(duan)
                    new = Duan.new(duan.index + 1, elements[0])
                    new.elements = elements
                    new.end = elements[-1].end
                    if double_relation(left, mid) is Direction.JumpUp:
                        duan.pre = new
                    ddp("    " * level, "顶分型终结", duan.pre is not None, elements[0].direction)
                    features = FeatureSequence.analysis(elements, new.direction)
                    if features:
                        new.features = [features[-1], None, None]
                        if len(features) > 1:
                            new.features = [features[-2], features[-1], None]
                    # Duan.append(duans, new, ZShandler)
                    if duan.end is new.start:
                        duans.append(new)
                        self.__push_duan_zs(new)
                    else:
                        raise ChanException("线段不连续", duan.elements[-1].end, new.elements[0].start)

            else:
                raise ChanException("未知的关系", relation)

        else:
            raise ChanException("未知的状态", state, lmr)

        duans[-1].check()

    def __pop_duan_zs(self, duan: Duan):
        zss = self._duan_zss
        if zss:
            last = zss[-1]
            if last.elements[-1] is duan:
                last.elements.pop()
                if len(last.elements) == 0:
                    zss.pop()
                    if zss:
                        last = zss[-1]
                        if last.elements[-1] is duan:
                            ddp("递归弹出")
                            self.__pop_duan_zs(duan)
            else:
                dp("中枢中没有此元素", duan)

    def __push_duan_zs(self, duan: Duan):
        zss = self._duan_zss
        new = ZhongShu(duan)
        duans = self._duans
        if not zss:
            zss.append(new)
            return
        zs = zss[-1]
        if len(zs.elements) >= 3:
            relation = double_relation(zs, duan)
            if relation in (Direction.JumpUp, Direction.JumpDown):
                if duan.done:
                    zss.append(new)
                else:
                    zs.append_element(duan)
            else:
                zs.append_element(duan)

        elif len(zs.elements) == 2:
            if double_relation(zs.elements[0], duan) in (Direction.JumpUp, Direction.JumpDown):
                # 这里需要判断走势
                zss.pop()
                zss.append(new)
            else:
                zs.append_element(duan)

        elif len(zs.elements) == 1:
            if zs.elements[0].index > 1:
                relation = double_relation(duans[zs.elements[0].index - 2], zs.elements[0])
                if (zs.elements[0].direction is Direction.Up and relation is Direction.Up) or (zs.elements[0].direction is Direction.Down and relation is Direction.Down):
                    zss.pop()
                    zss.append(new)
                else:
                    zs.append_element(duan)
            else:
                zs.append_element(duan)

        else:
            zss.pop()
            zss.append(new)

    def __push_bi_zs(self, bi: Bi):
        bis = self._bis
        zss = self._bi_zss
        new = ZhongShu(bi)
        if not zss:
            zss.append(new)
            return
        zs = zss[-1]

        if len(zs.elements) >= 3:
            relation = double_relation(zs, bi)
            if relation in (Direction.JumpUp, Direction.JumpDown):
                zss.append(new)

            else:
                zs.append_element(bi)
        elif len(zs.elements) == 2:
            if double_relation(zs.elements[0], bi) in (Direction.JumpUp, Direction.JumpDown):
                # 这里需要判断走势
                zss.pop()
                zss.append(new)
            else:
                zs.append_element(bi)

        elif len(zs.elements) == 1:
            if zs.elements[0].index > 1:
                relation = double_relation(bis[zs.elements[0].index - 2], zs.elements[0])
                if (zs.elements[0].direction is Direction.Up and relation is Direction.Up) or (zs.elements[0].direction is Direction.Down and relation is Direction.Down):
                    zss.pop()
                    zss.append(new)
                else:
                    zs.append_element(bi)
            else:
                zs.append_element(bi)

        else:
            zss.pop()
            zss.append(new)

    def __pop_bi_zs(self, obj: Bi):
        zss = self._bi_zss

        if not zss:
            return
        last = zss[-1]
        last.pop_element(obj)
        if last.last is None:
            zss.pop()
            if zss:
                last = zss[-1]
                if obj is last.last:
                    dp("递归删除中枢元素", obj)
                    self.__pop_bi_zs(obj)

    def process(self):
        self._duans.clear()
        self._bi_zss.clear()
        self._duan_zss.clear()
        self._zss.clear()
        for bi in self._bis:
            self.__push_duan(bi)

    def toCharts(self, path: str = "czsc.html", useReal=False):
        import echarts_plot  # czsc

        reload(echarts_plot)
        kline_pro = echarts_plot.kline_pro
        fx = [{"dt": fx.dt, "fx": fx.low if fx.shape is Shape.D else fx.high} for fx in self._fxs]
        bi = [{"dt": fx.dt, "bi": fx.low if fx.shape is Shape.D else fx.high} for fx in self._fxs]

        # xd = [{"dt": fx.dt, "xd": fx.low if fx.shape is Shape.D else fx.high} for fx in self.xd_fxs]

        xd = []
        mergers = []
        for duan in self._duans:
            xd.extend(duan.charts())
            left, mid, right = duan.features
            if left:
                if len(left) > 1:
                    mergers.append(left)
            if mid:
                if len(mid) > 1:
                    mergers.append(mid)
            if right:
                if len(right) > 1:
                    mergers.append(right)
            else:
                print("right is None")

        dzs = [zs.charts() for zs in self._duan_zss if len(zs.elements) >= 3]
        bzs = [zs.charts() for zs in self._bi_zss if len(zs.elements) >= 3]

        charts = kline_pro(
            [x.candleDict() for x in self._raws] if useReal else [x.candleDict() for x in self._news],
            fx=fx,
            bi=bi,
            xd=xd,
            mergers=mergers,
            bzs=bzs,
            dzs=dzs,
            title=self.symbol + "-" + str(self.freq / 60) + "分钟",
            width="100%",
            height="80%",
        )

        charts.render(path)
        return charts


class CZSCAnalyzer:
    def __init__(self, symbol: str, freq: int, freqs: List[int] = None):
        self.symbol = symbol
        self.freq = freq
        self.freqs = freqs or [freq]

        self.raws = RawBars([], freq, self.freqs)
        self.__analyzer = BaseAnalyzer(symbol, freq)
        self.__analyzer._raws = self.raws

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
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            index=index,
        )
        self.push(last)

    def push(self, k: RawBar):
        self.raws.push(k)

        try:
            self.__analyzer.push(self.raws[-1])
        except Exception as e:
            self.__analyzer.toCharts()
            # with open(f"{self.symbol}-{int(self._bars[0].dt.timestamp())}-{int(self._bars[-1].dt.timestamp())}.dat", "wb") as f:
            #    f.write(self.save_bytes())
            raise e

    @classmethod
    def load_bytes(cls, symbol: str, bytes_data: bytes, freq: int) -> "Self":
        size = struct.calcsize(">6d")
        obj = cls(symbol, freq)
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
        with open(f"{self.symbol}-{self.freq}-{int(self.__analyzer._raws[0].dt.timestamp())}-{int(self.__analyzer._raws[-1].dt.timestamp())}.dat", "wb") as f:
            f.write(self.save_bytes())

    @classmethod
    def load_file(cls, path: str) -> "Self":
        name = Path(path).name.split(".")[0]
        symbol, freq, s, e = name.split("-")
        with open(path, "rb") as f:
            dat = f.read()
            return cls.load_bytes(symbol, dat, int(freq))

    def toCharts(self, path: str = "czsc.html", useReal=False):
        self.__analyzer.toCharts(path=path, useReal=useReal)


class Bitstamp(CZSCAnalyzer):
    """ """

    def __init__(self, symbol: str, freq: Union[Freq, int, str], size: int = 0):
        if type(freq) is Freq:
            super().__init__(symbol, freq.value)
            self.freq: int = freq.value
        elif type(freq) is int:
            super().__init__(symbol, freq)
            self.freq: int = freq
        elif type(freq) is str:
            super().__init__(symbol, int(freq))
            self.freq: int = int(freq)
        else:
            raise

    def init(self, size):
        self.left_date_timestamp: int = int(datetime.now().timestamp() * 1000)
        left = int(self.left_date_timestamp / 1000) - self.freq * size
        if left < 0:
            raise ChanException
        _next = left
        while 1:
            data = self.ohlc(self.symbol, self.freq, _next, _next := _next + self.freq * 1000)
            if not data.get("data"):
                print(data)
                raise ChanException
            for bar in data["data"]["ohlc"]:
                try:
                    self.step(
                        bar["timestamp"],
                        bar["open"],
                        bar["high"],
                        bar["low"],
                        bar["close"],
                        bar["volume"],
                    )
                except ChanException as e:
                    # continue
                    raise e

            # start = int(data["data"]["ohlc"][0]["timestamp"])
            end = int(data["data"]["ohlc"][-1]["timestamp"])

            _next = end
            if len(data["data"]["ohlc"]) < 100:
                break

    @staticmethod
    def ohlc(pair: str, step: int, start: int, end: int, length: int = 1000) -> Dict:
        proxies = {
            "http": "http://127.0.0.1:11809",
            "https": "http://127.0.0.1:11809",
        }
        s = requests.Session()

        s.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",
            "content-type": "application/json",
        }
        url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/?step={step}&limit={length}&start={start}&end={end}"
        resp = s.get(url, timeout=5, proxies=proxies)
        json = resp.json()
        # print(json)
        return json


def main():
    bitstamp = Bitstamp("btcusd", freq=Freq.m5, size=3500)
    bitstamp.init(8000)
    bitstamp.toCharts()
    return bitstamp


if __name__ == "__main__":
    bit = main()
    bit.save_file()
    # bit = Bitstamp.load_file("btcusd-14400-1600416000-1715601600.dat")
    # bit.toCharts()
