"""
Implementation of the TraX sever. This module provides implementation
of the server side of the protocol and is therefore meant to be used in the
tracker.
"""

import collections
import logging as log
import os
import socket
import sys

import trax.image
import trax.region

from . import TraXError, MessageType
from .message import MessageParser

DEFAULT_HOST = '127.0.0.1'

TRAX_VERSION = 1


class Request(collections.namedtuple('Request', ['type', 'image', 'region', 'parameters'])):
    """ A container class for client requests. Contains fileds type, image, region and parameters. """


class Server(MessageParser):
    """ TraX server implementation class."""

    def __init__(self, options, verbose=False):
        """ Constructor.

            :param bool verbose: if True display log info
        """
        if verbose:
            log.basicConfig(
                format="%(levelname)s: %(message)s", level=log.DEBUG)
        else:
            log.basicConfig(format="%(levelname)s: %(message)s")

        self.options = options

        if "TRAX_SOCKET" in os.environ:

            port = int(os.environ.get("TRAX_SOCKET"))

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            log.info('Socket created')
            # Connect to localhost
            try:
                self.socket.connect((DEFAULT_HOST, port))
                log.info('Server connected to client')
            except socket.error as msg:
                log.error(
                    'Connection failed. Error Code: {}\nMessage: {}'.format(str(msg[0]), msg[1]))
                raise TraXError(
                    "Unable to connect to client: {}\nMessage: {}".format(str(msg[0]), msg[1]))

            super(Server, self).__init__(
                os.fdopen(self.socket.fileno(), 'r'), os.fdopen(self.socket.fileno(), 'w'))

        else:
            fin = sys.stdin
            fout = sys.stdout

            env_in = int(os.environ.get("TRAX_IN", "-1"))
            env_out = int(os.environ.get("TRAX_OUT", "-1"))

            if env_in > 0:
                print('Using file stream {} for input and {} for output.'.format(env_in, env_out))

            super(Server, self).__init__(
                os.fdopen(env_in, 'r') if env_in > 0 else fin,
                os.fdopen(env_out, 'w') if env_out > 0 else fout
            )

        self._setup()

    def _setup(self):
        """ Send hello message with capabilities to the TraX client """

        properties = {"trax.{}".format(prop): getattr(
            self.options, prop) for prop in self.options.__dict__.keys()}
        self._write_message(MessageType.HELLO, [], properties)
        return

    def wait(self):
        """ Wait for client message request. Recognize it and parse them when received .

            :returns: A request structure
            :rtype: trax.server.Request
        """
        print("Trax server reading message...")
        message = self._read_message()
        print("Trax received message of type {}!".format(message.type))
        if message.type == None:
            return Request(MessageType.ERROR, None, None, None)

        elif message.type == MessageType.QUIT and len(message.arguments) == 0:
            log.info('Received quit message from client.')
            return Request(message.type, None, None, message.parameters)

        elif message.type == MessageType.INITIALIZE and len(message.arguments) == 2:
            log.info('Received initialize message.')

            image = trax.image.parse(message.arguments[0])
            region = trax.region.parse(message.arguments[1])
            return Request(message.type, image, region, message.parameters)

        elif message.type == MessageType.FRAME and len(message.arguments) == 1:
            log.info('Received frame message.')

            image = trax.image.parse(message.arguments[0])
            return Request(message.type, image, None, message.parameters)

        else:
            return Request(MessageType.ERROR, None, None, None)

    def status(self, region, properties=None):
        """ Reply to client with a status region and optional properties.


            :param trax.region.Region region: Resulting region object.
            :param dict properties: Optional arguments as a dictionary.
        """
        assert (isinstance(region, trax.region.Region))
        self._write_message(MessageType.STATUS, [region], properties)

    def __enter__(self):
        """ To support instantiation with 'with' statement. """
        return self

    def __exit__(self, *args, **kwargs):
        """ Destructor used by 'with' statement. """
        self.quit()

    def quit(self):
        """ Sends quit message and end terminates communication. """
        try:
            self._close()
            if hasattr(self, 'socket'):
                self.socket.close()
        except IOError:
            pass


class ServerOptions(object):
    """ TraX server options """

    def __init__(self, region, image, name=None, identifier=None):
        """ Constructor for server configuration

            :param str name: name of the tracker
            :param str identifier: identifier of the current implementation
            :param str region: region format.
            :param str image: image format.
        """

        if not type(region) == list:
            region = [region]
        if not type(image) == list:
            image = [image]

        # other formats not implemented yet
        for r in region:
            assert (r in [trax.region.RECTANGLE, trax.region.POLYGON])
        for i in image:
            assert (i in [trax.image.PATH, trax.image.URL, trax.image.MEMORY, trax.image.BUFFER])

        if name:
            self.name = name
        if identifier:
            self.identifier = identifier
        self.region = ";".join(region)
        self.image = ";".join(image)
        self.version = TRAX_VERSION
