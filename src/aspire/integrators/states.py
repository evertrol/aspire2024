__all__ = ["OrbitState"]


class OrbitState:
    """Container to hold the body position and velocity"""

    def __init__(self, x, y, u, v):
        self.x = x
        self.y = y
        self.u = u
        self.v = v

    def __add__(self, other):
        return OrbitState(
            self.x + other.x, self.y + other.y, self.u + other.u, self.v + other.v
        )

    def __sub__(self, other):
        return OrbitState(
            self.x - other.x, self.y - other.y, self.u - other.u, self.v - other.v
        )

    def __mul__(self, other):
        return OrbitState(
            other * self.x, other * self.y, other * self.u, other * self.v
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return f"{self.x:10.6f} {self.y:10.6f} {self.u:10.6f} {self.v:10.6f}"
