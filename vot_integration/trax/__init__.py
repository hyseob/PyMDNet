"""
Implementation of the TraX protocol. The current implementation is written in pure Python and is therefore a bit slow.
"""


class MessageType(object):
    """ The message type container class """
    ERROR = "error"
    HELLO = "hello"
    INITIALIZE = "initialize"
    FRAME = "frame"
    QUIT = "quit"
    STATUS = "status"


class TraXError(RuntimeError):
    """ A protocol error class """
    pass
