"""
\file vot.py

@brief Python utility functions for VOT integration

@author Luka Cehovin, Alessio Dore

@date 2016

"""

import collections
import copy

try:
    from trax import server
    from trax.region import *
    from trax.image import *

    TRAX = True
except ImportError:
    TRAX = False

# Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])


# Polygon = collections.namedtuple('Polygon', ['points'])


def parse_region(string):
    tokens = list(map(float, string.split(',')))
    if len(tokens) == 4:
        return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
    elif len(tokens) % 2 == 0 and len(tokens) > 4:
        return Polygon([Point(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)])
    return None


def encode_region(region):
    if isinstance(region, Polygon):
        return ','.join(['{},{}'.format(p.x, p.y) for p in region.points])
    elif isinstance(region, Rectangle):
        return '{},{},{},{}'.format(region.x, region.y, region.width, region.height)
    else:
        return ""


def convert_region(region, to):
    if to == 'rectangle':

        if isinstance(region, Rectangle):
            return copy.copy(region)
        elif isinstance(region, Polygon):
            top = sys.float_info.max
            bottom = sys.float_info.min
            left = sys.float_info.max
            right = sys.float_info.min

            for point in region.points:
                top = min(top, point.y)
                bottom = max(bottom, point.y)
                left = min(left, point.x)
                right = max(right, point.x)

            return Rectangle(left, top, right - left, bottom - top)

        else:
            return None
    if to == 'polygon':

        if isinstance(region, Rectangle):
            points = [(region.x, region.y),
                      (region.x + region.width, region.y),
                      (region.x + region.width, region.y + region.height),
                      (region.x, region.y + region.height)]
            return Polygon(points)

        elif isinstance(region, Polygon):
            return copy.copy(region)
        else:
            return None

    return None


class VOT(object):
    """ Base class for Python VOT integration """

    def __init__(self, region_format):
        """ Constructor

        Args:
            region_format: Region format options
        """
        assert (region_format in ['rectangle', 'polygon'])
        if TRAX:
            options = server.ServerOptions(region_format, PATH)
            self._trax = server.Server(options)

            request = self._trax.wait()
            assert (request.type == 'initialize')
            if request.region.type == 'polygon':
                self._region = Polygon([Point(x[0], x[1]) for x in request.region.points])
            else:
                self._region = Rectangle(request.region.x, request.region.y, request.region.width,
                                         request.region.height)
            self._image = str(request.image)
            self._trax.status(request.region)
        else:
            self._files = [x.strip('\n') for x in open('images.txt', 'r').readlines()]
            self._frame = 0
            self._region = convert_region(parse_region(open('region.txt', 'r').readline()), region_format)
            self._result = []

    def region(self):
        """
        Send configuration message to the client and receive the initialization
        region and the path of the first image

        Returns:
            initialization region
        """

        return self._region

    def report(self, region, confidence=0):
        """
        Report the tracking results to the client

        Arguments:
            :param region: region for the frame
            :param confidence: confidence of the prediction
        """
        assert (isinstance(region, Rectangle) or isinstance(region, Polygon))
        if TRAX:
            if isinstance(region, Polygon):
                tregion = Polygon([(x.x, x.y) for x in region.points])
            else:
                tregion = Rectangle(region.x, region.y, region.width, region.height)
            self._trax.status(tregion, {"confidence": confidence})
        else:
            self._result.append(region)
            self._frame += 1

    def frame(self):
        """
        Get a frame (image path) from client

        Returns:
            absolute path of the image
        """
        if TRAX:
            if hasattr(self, "_image"):
                image = str(self._image)
                del self._image
                return image

            print("Waiting for Trax...")
            request = self._trax.wait()

            print("Trax returned request of type {}!".format(request.type))
            if request.type == 'frame':
                return str(request.image)
            else:
                return None

        else:
            if self._frame >= len(self._files):
                return None
            return self._files[self._frame]

    def quit(self):
        if TRAX:
            self._trax.quit()
        elif hasattr(self, '_result'):
            with open('output.txt', 'w') as f:
                for r in self._result:
                    f.write(encode_region(r))
                    f.write('\n')

    def __del__(self):
        self.quit()
