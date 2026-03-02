class Dimension(str):
  def __class_getitem__(cls, item):
    if isinstance(item, tuple):
      return cls(" ".join(cls(i) for i in item))
    return cls(item)

  def __add__(self, other):
    return self.__class__(f"({self}+{other})")

  def __radd__(self, other):
    # WARNING: This is only called if 'other' is NOT a string.
    # "String" + Dimension("N") results in concatenation, not this method.
    return self.__class__(f"({other}+{self})")

  def __sub__(self, other):
    return self.__class__(f"({self}-{other})")

  def __rsub__(self, other):
    return self.__class__(f"({other}-{self})")

  def __mul__(self, other):
    return self.__class__(f"({self}*{other})")

  def __rmul__(self, other):
    return self.__class__(f"({other}*{self})")

  def __truediv__(self, other):
    return self.__class__(f"({self}/{other})")

  def __rtruediv__(self, other):
    return self.__class__(f"({other}/{self})")

  def __floordiv__(self, other):
    return self.__class__(f"({self}//{other})")

  def __rfloordiv__(self, other):
    return self.__class__(f"({other}//{self})")

  def __pow__(self, other):
    return self.__class__(f"({self}**{other})")

  def __rpow__(self, other):
    return self.__class__(f"({other}**{self})")

  def __mod__(self, other):
    return self.__class__(f"({self}%{other})")

  def __rmod__(self, other):
    return self.__class__(f"({other}%{self})")

  def __neg__(self):
    return self.__class__(f"-{self}")

  def __abs__(self):
    return self.__class__(f"abs({self})")


# Arrays
Scalar = Dimension[""]
B = Dimension["B"]
N = Dimension["N"]
P = Dimension["P"]
L = Dimension["L"]
C = Dimension["C"]
H = Dimension["H"]
W = Dimension["W"]
sB = Dimension["*B"]
sN = Dimension["*N"]
sL = Dimension["*L"]
sC = Dimension["*C"]
hN = Dimension["#N"]
hL = Dimension["#L"]
hC = Dimension["#C"]
_B = Dimension["_B"]
_N = Dimension["_N"]
_L = Dimension["_L"]
_C = Dimension["_C"]
__ = Dimension["_"]
Any = Dimension["..."]

# Trees
T = Dimension["T"]
