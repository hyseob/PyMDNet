"""
Region description classes.
"""

import sys
from functools import reduce

if sys.version_info > (3, 0):
    xrange = range

SPECIAL = "special"
""" Constant for special region type """

RECTANGLE = "rectangle"
""" Constant for rectangle region type """

POLYGON = "polygon"
""" Constant for polygon region type """


def convert(region, to):
    """
    Perform conversion from one region type to another (if possible).

    :param Region region: original region
    :param str to: type of desired result
    :result: converter region or None if conversion is not possible
    """

    if to == RECTANGLE:

        if isinstance(region, Rectangle):
            return region.copy()
        elif isinstance(region, Polygon):
            top = sys.float_info.min
            bottom = sys.float_info.max
            left = sys.float_info.min
            right = sys.float_info.max

            for point in region.points:
                top = min(top, point[1])
                bottom = max(bottom, point[1])
                left = min(left, point[0])
                right = max(right, point[0])

            return Rectangle(left, top, right - left, bottom - top)

        else:
            return None
    if to == POLYGON:

        if isinstance(region, Rectangle):
            points = [(region.x, region.y), (region.x + region.width, region.y),
                      (region.x + region.width, region.y + region.height), (region.x, region.y + region.height)]
            return Polygon(points)

        elif isinstance(region, Polygon):
            return region.copy()
        else:
            return None

    elif to == SPECIAL:
        if isinstance(region, Special):
            return region.copy()
        else:
            return Special()

    return None


def parse(string):
    tokens = list(map(float, string.split(',')))
    if len(tokens) == 1:
        return Special(tokens[0])
    if len(tokens) == 4:
        return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
    elif len(tokens) % 2 == 0 and len(tokens) > 4:
        return Polygon([(tokens[i], tokens[i + 1]) for i in xrange(0, len(tokens), 2)])
    return None


class Region(object):
    """
    Base class for all region containers

    :var type: type of the region
    """

    def __init__(self, type):
        self.type = type


class Special(Region):
    """
    Special region

    :var code: Code value
    """

    def __init__(self, code):
        """ Constructor

        :param code: Special code
        """
        super(Special, self).__init__(SPECIAL)
        self.code = int(code)

    def __str__(self):
        """ Create string from class to send to client """
        return '{}'.format(self.code)


class Rectangle(Region):
    """
    Rectangle region

    :var x: top left x coord of the rectangle region
    :var float y: top left y coord of the rectangle region
    :var float w: width of the rectangle region
    :var float h: height of the rectangle region
    """

    def __init__(self, x=0, y=0, width=0, height=0):
        """ Constructor

            :param float x: top left x coord of the rectangle region
            :param float y: top left y coord of the rectangle region
            :param float w: width of the rectangle region
            :param float h: height of the rectangle region
        """
        super(Rectangle, self).__init__(RECTANGLE)
        self.x, self.y, self.width, self.height = x, y, width, height

    def __str__(self):
        """ Create string from class to send to client """
        return '{},{},{},{}'.format(self.x, self.y, self.width, self.height)


class Polygon(Region):
    """
    Polygon region

    :var list points: List of points as tuples [(x1,y1), (x2,y2),...,(xN,yN)]
    :var int count: number of points
    """

    def __init__(self, points):
        """
        Constructor

        :param list points: List of points as tuples [(x1,y1), (x2,y2),...,(xN,yN)]
        """
        super(Polygon, self).__init__(POLYGON)
        assert (isinstance(points, list))
        # do not allow empty list
        assert (len(points) > 0)
        assert (reduce(lambda x, y: x and y, [isinstance(p, tuple) for p in points]))
        self.count = len(points)
        self.points = points

    def __str__(self):
        """ Create string from class to send to client """
        return ','.join(['{},{}'.format(p[0], p[1]) for p in self.points])
