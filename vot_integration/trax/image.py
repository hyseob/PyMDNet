"""
Image description classes. 
"""

import base64
import re

URL_PATTERN = re.compile('^(?:[a-z-]+)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
FILE_PATTERN = re.compile('^file://(?P<file>.+)')
DATA_PATTERN = re.compile('^data:(?P<format>[a-z/0-9]+);(?:[a-z/0-9]+;)?')
IMAGE_PATTERN = re.compile('^image:(?P<width>[0-9]+);(?P<height>[0-9]+);(?P<format>[a-z/0-9]+);')

PATH = "path"
""" Constant for file path image """

URL = "url"
""" Constant for remote or local URL image """

BUFFER = "buffer"
""" Constant for encoded memory buffer image """

MEMORY = "memory"
""" Constant for raw memory image """


def parse(string):
    """ Parses string image representation to one of the containers """

    match = DATA_PATTERN.match(string)
    if match:
        format = match.group("format")

        if format not in ['image/jpeg', 'image/png']:
            return None
        return BufferImage(base64.b64decode(string[match.end():]), format)

    else:
        match = IMAGE_PATTERN.match(string)
        if match:
            import numpy

            IMAGE_SETUPS = {
                'gray8': (numpy.uint8, 1),
                'gray16': (numpy.uint16, 1),
                'rgb': (numpy.uint8, 3),
            }

            width = int(match.group("width"))
            height = int(match.group("height"))
            format = match.group("format")

            if not format in ['gray8', 'gray16', 'rgb']:
                return None

            image = numpy.fromstring(base64.b64decode(string[match.end():]), dtype=IMAGE_SETUPS[format][0])

            return MemoryImage(image.reshape(height, width, IMAGE_SETUPS[format][1]))
        else:
            match = FILE_PATTERN.match(string)
            if match:
                return FileImage(match.group("file"))
            else:
                match = URL_PATTERN.match(string)
                if match:
                    return URLImage(string)

    return None


class Image(object):
    """ 
    Base class for all image containers

    :var type: Type constant for the image
    """

    def __init__(self, type):
        self.type = type


class FileImage(Image):
    """ 
    Image saved in a local file 

    :var path: Path to the image file
    """

    def __init__(self, path=None):
        super(FileImage, self).__init__(PATH)
        self.path = path

    def __str__(self):
        """ Create string from image data """
        return self.path


class URLImage(Image):
    """ 
    Image saved in a local or remote resource 

    :var url: URL of the image
    """

    """ Contains URL as string """

    def __init__(self, url=None):
        super(URLImage, self).__init__(URL)
        self.url = url

    def __str__(self):
        """ Create string from image data """
        return self.url


class MemoryImage(Image):
    """ Image saved in memory as a numpy array """

    def __init__(self, image):
        super(MemoryImage, self).__init__(MEMORY)
        self.image = image

    def __str__(self):
        """ Create string from image data """
        width = self.image.shape[1]
        height = self.image.shape[0]
        format = "test"
        return "image:{};{};{};{}".format(width, height, format, base64.b64encode(self.image.tobytes()))


class BufferImage(Image):
    """ Image encoded in a memory buffer stored in JPEG or PNG file format """

    def __init__(self, data=None, format="unknown"):
        super(BufferImage, self).__init__(BUFFER)
        self.data = data
        self.format = format

    def __str__(self):
        """ Create string from image data """
        return "data:{};base64;{}".format(self.format, base64.b64encode(self.data))
