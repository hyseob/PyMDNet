import collections
import re
import sys

from . import MessageType, TraXError

if sys.version_info > (3, 0):
    xrange = range

TRAX_PREFIX = '@@TRAX:'

TRAX_BUFFER_SIZE = 128

PARSE_STATE_TYPE, PARSE_STATE_SPACE_EXPECT, PARSE_STATE_SPACE, PARSE_STATE_UNQUOTED_KEY, PARSE_STATE_UNQUOTED_VALUE, PARSE_STATE_UNQUOTED_ESCAPE_KEY, PARSE_STATE_UNQUOTED_ESCAPE_VALUE, PARSE_STATE_QUOTED_KEY, PARSE_STATE_QUOTED_VALUE, PARSE_STATE_QUOTED_ESCAPE_KEY, PARSE_STATE_QUOTED_ESCAPE_VALUE = xrange(
    0, 11)

PARSE_STATE_PASS = 100
PARSE_MAX_KEY_LENGTH = 16

VALID_KEY_PATTERN = re.compile("^[0-9a-zA-Z\\._]+$")

Message = collections.namedtuple('Message', ['type', 'arguments', 'parameters'])


def _isValidKey(key):
    if len(key) < 1 or len(key) > PARSE_MAX_KEY_LENGTH:
        return False

    return VALID_KEY_PATTERN.match(key) is not None


def _parseMessageType(data):
    assert (data.lower() in [MessageType.HELLO, MessageType.INITIALIZE, MessageType.FRAME, MessageType.QUIT,
                             MessageType.STATUS])
    return data.lower()


class MessageParser(object):
    def __init__(self, fin, fout):
        self._fin = fin
        self._fout = fout

        self._opened = True

    def _close(self):
        self._opened = False

    def _read_message(self):
        """ Read socket message and parse it

        Returns:
            msgArgs: list of message arguments
        """
        assert self._opened

        keyBuffer = ""
        valueBuffer = ""

        complete = False

        state = -len(TRAX_PREFIX)
        while not complete:
            val = self._fin.read(1)
            if val == -1:
                if message is None:
                    break
                char = '\n'
                complete = True
            else:
                char = str(val)
            if state == PARSE_STATE_TYPE:  # Parsing message type
                try:
                    if char.isalnum():
                        keyBuffer += char
                    elif char == ' ':
                        message = Message(_parseMessageType(keyBuffer.lower()), [], {})
                        state = PARSE_STATE_SPACE
                        keyBuffer = ""
                        valueBuffer = ""
                    elif char == '\n':
                        message = Message(_parseMessageType(keyBuffer.lower()), [], {})
                        keyBuffer = ""
                        valueBuffer = ""
                        complete = True
                    else:
                        state = PARSE_STATE_PASS
                        keyBuffer = ""
                except (TraXError):
                    state = PARSE_STATE_PASS
                    message = None
                    keyBuffer = ""
            elif state == PARSE_STATE_SPACE_EXPECT:
                if char == ' ':
                    state = PARSE_STATE_SPACE
                elif char == '\n':
                    complete = True
                else:
                    message = None
                    state = PARSE_STATE_PASS
                    keyBuffer = char
                    valueBuffer = ""
            elif state == PARSE_STATE_SPACE:
                if char == ' ' or char == '\r':
                    # Do nothing
                    pass
                elif char == '\n':
                    complete = True
                elif char == '"':
                    state = PARSE_STATE_QUOTED_KEY
                    keyBuffer = ""
                    valueBuffer = ""
                else:
                    state = PARSE_STATE_UNQUOTED_KEY
                    keyBuffer = char
                    valueBuffer = ""

            elif state == PARSE_STATE_UNQUOTED_KEY:
                if char == '\\':
                    state = PARSE_STATE_UNQUOTED_ESCAPE_KEY
                elif char == '\n':  # append arg and finalize
                    message.arguments.append(keyBuffer)
                    complete = True
                elif char == ' ':  # append arg and move on
                    message.arguments.append(keyBuffer)
                    state = PARSE_STATE_SPACE
                    keyBuffer = ""
                elif char == '=':  # we have a kwarg
                    if _isValidKey(keyBuffer):
                        state = PARSE_STATE_UNQUOTED_VALUE
                    else:
                        keyBuffer += char
                else:
                    keyBuffer += char
            elif state == PARSE_STATE_UNQUOTED_VALUE:
                if char == '\\':
                    state = PARSE_STATE_UNQUOTED_ESCAPE_VALUE
                elif char == ' ':
                    message.parameters[keyBuffer] = valueBuffer
                    state = PARSE_STATE_SPACE
                    keyBuffer = ""
                    valueBuffer = ""
                elif char == '\n':
                    message.parameters[keyBuffer] = valueBuffer
                    complete = True
                    keyBuffer = ""
                    valueBuffer = ""
                else:
                    valueBuffer += char

            elif state == PARSE_STATE_UNQUOTED_ESCAPE_KEY:

                if char == 'n':
                    keyBuffer += '\n'
                    state = PARSE_STATE_UNQUOTED_KEY
                elif char != '\n':
                    keyBuffer += char
                    state = PARSE_STATE_UNQUOTED_KEY
                else:
                    state = PARSE_STATE_PASS
                    message = None
                    keyBuffer = ""
                    valueBuffer = ""

            elif state == PARSE_STATE_UNQUOTED_ESCAPE_VALUE:
                if char == 'n':
                    valueBuffer += '\n'
                    state = PARSE_STATE_UNQUOTED_VALUE
                elif char != '\n':
                    valueBuffer += char
                    state = PARSE_STATE_UNQUOTED_VALUE
                else:
                    state = PARSE_STATE_PASS
                    message = None
                    keyBuffer = ""
                    valueBuffer = ""

            elif state == PARSE_STATE_QUOTED_KEY:
                if char == '\\':
                    state = PARSE_STATE_QUOTED_ESCAPE_KEY
                elif char == '"':  # append arg and move on
                    message.arguments.append(keyBuffer)
                    state = PARSE_STATE_SPACE_EXPECT
                elif char == '=':  # we have a kwarg
                    if _isValidKey(keyBuffer):
                        state = PARSE_STATE_QUOTED_VALUE
                    else:
                        keyBuffer += char
                else:
                    keyBuffer += char
            elif state == PARSE_STATE_QUOTED_VALUE:
                if char == '\\':
                    state = PARSE_STATE_QUOTED_ESCAPE_VALUE
                elif char == '"':
                    message.parameters[keyBuffer] = valueBuffer
                    state = PARSE_STATE_SPACE_EXPECT
                    keyBuffer = ""
                    valueBuffer = ""
                else:
                    valueBuffer += char

            elif state == PARSE_STATE_QUOTED_ESCAPE_KEY:
                if char == 'n':
                    keyBuffer += '\n'
                    state = PARSE_STATE_QUOTED_KEY
                elif char != '\n':
                    keyBuffer += char
                    state = PARSE_STATE_QUOTED_KEY
                else:
                    state = PARSE_STATE_PASS
                    message = None
                    keyBuffer = ""
                    valueBuffer = ""

            elif state == PARSE_STATE_QUOTED_ESCAPE_VALUE:
                if char == 'n':
                    valueBuffer += '\n'
                    state = PARSE_STATE_QUOTED_VALUE
                elif char != '\n':
                    valueBuffer += char
                    state = PARSE_STATE_QUOTED_VALUE
                else:
                    state = PARSE_STATE_PASS
                    message = None
                    keyBuffer = ""
                    valueBuffer = ""
            elif state == PARSE_STATE_PASS:
                if char == '\n':
                    state = -len(TRAX_PREFIX)
            else:  # Parsing prefix
                if state < 0:
                    if char == TRAX_PREFIX[len(TRAX_PREFIX) + state]:
                        # When done, go to type parsing
                        state += 1
                    else:  # Not a message
                        state = -len(TRAX_PREFIX) if char == '\n' else PARSE_STATE_PASS
        return message

    def _write_message(self, mtype, arguments, properties):
        """ Create the message string and send it

        Args:
            mtype: message type identifier
            arguments: message arguments
            properties: optional arguments. Format: "key:value"
        """
        assert (self._opened)
        assert (isinstance(arguments, list))
        assert (isinstance(properties, dict) or properties is None)
        assert (mtype in [MessageType.HELLO, MessageType.INITIALIZE, MessageType.FRAME, MessageType.QUIT,
                          MessageType.STATUS])

        self._fout.write(TRAX_PREFIX)
        self._fout.write(mtype)

        for arg in arguments:
            self._fout.write(" ")
            if not isinstance(arg, str):
                arg = str(arg)
            arg = arg.replace("\"", "\\\"").replace("\\", "\\\\").replace("\n", "\\n")
            self._fout.write('\"' + arg + '\"')

        # optional arguments
        if properties:
            for k, v in properties.items():
                self._fout.write(" ")
                arg = "{}={}".format(k, str(v)).replace("\"", "\\\"").replace("\\", "\\\\").replace("\n", "\\n")
                self._fout.write('\"' + arg + '\"')

        self._fout.write("\n")
        self._fout.flush()
