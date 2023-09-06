# from collections import namedtuple
# Resolution = namedtuple('Resolution', ['width', 'height'])

from dataclasses import dataclass

@dataclass(frozen=True)
class Resolution:
    width: int
    height: int

    def __getitem__(self, index):
        if index == 0:
            return self.width
        elif index == 1:
            return self.height
        else:
            raise IndexError('Resolution only has width and height, tried to access index outside of 0 or 1')
    def __len__(self):
        return 2

    def __iter__(self):
        return iter((self.width, self.height))
