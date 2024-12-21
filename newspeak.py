import argparse
import sys
from pathlib import Path
import enum
from typing import Tuple, List, Dict, Optional, Union, Iterator, Set, cast, Any
import re
import time
import threading


class NotCorrectStructureException(Exception):
    pass


class StatesOfGame(enum.Enum):
    FINDING_COMMAND: int = enum.auto()

    EXECUTING_COMMAND_PART: int = enum.auto()  # placeholder

    DELETING_COMMAND: int = enum.auto()
    FINISHED: int = enum.auto()

    ERROR: int = enum.auto()


class LanguageKeywords(enum.Enum):
    FIND: int = enum.auto()

    REMOVE: int = enum.auto()
    DELETE: int = enum.auto()
    REPLACE: int = enum.auto()
    SEEK: int = enum.auto()
    INSERT: int = enum.auto()
    APPEND: int = enum.auto()

    COMMENT: int = enum.auto()

    ONCE: int = enum.auto()


KEYWORDS_WITH_STR_ARG: Set[LanguageKeywords] = {
    LanguageKeywords.FIND,
    LanguageKeywords.REMOVE,
    LanguageKeywords.REPLACE,
    LanguageKeywords.INSERT,
    LanguageKeywords.APPEND,
    LanguageKeywords.COMMENT,
}


KEYWORDS_WITH_0_ARGS: Set[LanguageKeywords] = {
    LanguageKeywords.DELETE,
    LanguageKeywords.ONCE,
}


class SEEK_ARGS(enum.Enum):
    SOL: int = enum.auto()  # Start of line
    EOL: int = enum.auto()  # End of line
    SOF: int = enum.auto()  # Start of file
    EOF: int = enum.auto()  # End of file


class WrongKeywordException(Exception):
    pass


def parse_enum(string: str, enum_type: type[enum.Enum]) -> enum.Enum:
    string = string.strip()
    for member in enum_type:
        if member.name == string:
            return member

    raise WrongKeywordException()


def parse_SEEK_arg(input: Union[str, SEEK_ARGS]) -> Union[str, SEEK_ARGS]:
    if isinstance(input, SEEK_ARGS):
        return input

    try:
        return cast(SEEK_ARGS, parse_enum(input, SEEK_ARGS))
    except:
        _ = int(input)
        return input


def get_first_encountered_braces(text: str) -> Tuple[str, str, str]:
    braces_nesting: int = 0
    prev_char: str = ""
    return_text: List[str] = []

    starting_brace_pos: int = -1

    for idx, char in enumerate(text):
        if char == "{" and prev_char != "\\":
            braces_nesting += 1

            if starting_brace_pos < 0:
                starting_brace_pos = idx

        if braces_nesting > 0:
            return_text.append(char)

        if char == "}" and prev_char != "\\":
            braces_nesting -= 1

        if braces_nesting == 0 and return_text:
            return (
                text[:starting_brace_pos],
                "".join(return_text),
                text[idx + 1 :],
            )

        if braces_nesting < 0:
            raise NotCorrectStructureException()

        prev_char = char

    if braces_nesting == 0:
        return text, "", ""
    else:
        raise NotCorrectStructureException()


def find_char_backwards(in_str: str, starting_from_idx: int, ch: str):
    starting_from_idx = min(starting_from_idx, len(in_str) - 1)

    while starting_from_idx >= 0:
        ch_in_str: str = in_str[starting_from_idx]
        if ch_in_str == ch:
            return starting_from_idx
        starting_from_idx -= 1

    return 0


def find_in_text_with_protection_from_braces(
    text: str, searching_for_string: str, beg: int, end: int
) -> int:
    assert searching_for_string != ""
    assert text != ""

    if contains_unescaped_special_chars(searching_for_string):
        return text.index(searching_for_string, beg, end)

    escaped_text = re.sub(r"\\([{}])", r"\0\0", text)
    pattern = re.compile(re.escape(searching_for_string))

    for match in pattern.finditer(escaped_text):
        if match.start() < beg or match.end() > end:
            continue

        if encountered_unmatched_unprotected_brace(escaped_text, match.end()):
            continue

        if encountered_unmatched_unproteted_brace_backwards(escaped_text, match.end()):
            continue

        return match.start()

    raise ValueError()


def encountered_unmatched_unprotected_brace(text: str, starting_from: int) -> bool:
    previous_char: str = ""
    brace_nesting: int = 0
    for char in text[starting_from:]:
        if char == "}" and previous_char != "\\":
            if not brace_nesting:
                return True
            brace_nesting -= 1
        if char == "{" and previous_char != "\\":
            brace_nesting += 1
        previous_char = char
    return False


def encountered_unmatched_unproteted_brace_backwards(
    text: str, starting_from: int
) -> bool:
    brace_nesting: int = 0
    for char_idx in range(starting_from - 1, -1, -1):
        char: str = text[char_idx]
        one_before_char: str = ""
        if char_idx > 0:
            one_before_char = text[char_idx - 1]

        if one_before_char != "\\":
            if char == "{":
                if not brace_nesting:
                    return True
                brace_nesting += 1
            if char == "}":
                brace_nesting -= 1

    return False


def contains_unescaped_special_chars(text: str) -> bool:
    prev_char: str = ""
    for char in text:
        if char == "|" or char == "{" or char == "}":
            if prev_char != "\\":
                return True

        prev_char = char

    return False


def unescape_those_unprotected_by_braces(input: str) -> str:
    braces_nesting: int = 0
    previous_char: str = ""

    to_output: List[str] = []

    for char in input:
        if char == "{" and previous_char != "\\":
            braces_nesting += 1
        if char == "}" and previous_char != "\\":
            braces_nesting -= 1

        if char in "}{|" and previous_char == "\\" and braces_nesting == 0:
            to_output.pop()

        to_output.append(char)
        previous_char = char

    return "".join(to_output)


def remove_newlines_inside_commands(text: str) -> str:
    assert text != ""

    escaped_text = re.sub(r"\\([{}])", r"\0\0", text)
    pattern = re.compile("[\n\t]")

    to_remove: list[int] = []

    for match in pattern.finditer(escaped_text):
        if encountered_unmatched_unprotected_brace(
            escaped_text, match.start()
        ) or encountered_unmatched_unproteted_brace_backwards(
            escaped_text, match.end()
        ):
            to_remove.append(match.start())

    acccumulated_string: list[str] = []
    last_removed_idx: int = 0
    for to_remove_idx in to_remove:
        acccumulated_string.append(text[last_removed_idx:to_remove_idx])
        last_removed_idx = to_remove_idx + 1

    acccumulated_string.append(text[last_removed_idx : len(text)])
    return "".join(acccumulated_string)


def unpack_command(s: str) -> list:
    lst: list = []

    assert s == get_first_encountered_braces(s)[1]

    s = s[1:-1]  # stripping {}

    while True:
        before_part, command, after_part = get_first_encountered_braces(s)

        if before_part != "":
            lst.append(before_part)

        if command != "":
            lst.append([command])

        if after_part == "":
            return lst
        else:
            s = after_part


class ParsedCommand:
    def __init__(self) -> None:
        self.parts: List[Union[LanguageKeywords, str, SEEK_ARGS]] = []  # TODO

    def __repr__(self) -> str:
        lst: List[str] = ["{"]

        for part in self.parts:
            if isinstance(part, LanguageKeywords) or isinstance(part, SEEK_ARGS):
                lst.append(f"[{part.name}]")
            else:
                lst.append(part)
            lst.append("|")

        if lst[-1] == "|":
            lst.pop(-1)

        lst.append("}")

        return "".join(lst)


def parse_unpacked(unpacked: list) -> ParsedCommand:
    def unpacked_iterator() -> Iterator[str]:
        for element in unpacked:
            if isinstance(element, str):
                for char in element:
                    yield char
            elif isinstance(element, list):
                yield element[0]
            else:
                raise NotImplementedError()

    parts: List[Union[LanguageKeywords, str, SEEK_ARGS]] = []

    def append_to_parts(text: str) -> None:
        try:
            keyword: LanguageKeywords = cast(
                LanguageKeywords, parse_enum(text, LanguageKeywords)
            )
            parts.append(keyword)
        except:
            try:
                move_arg: Union[str, SEEK_ARGS] = parse_SEEK_arg(text)
                parts.append(move_arg)
            except:
                parts.append(text)

    accumulated_text: List[str] = []
    previous_char_or_braces: str = ""
    for char_or_braces in unpacked_iterator():
        if char_or_braces == "|" and previous_char_or_braces != "\\":
            append_to_parts("".join(accumulated_text))
            accumulated_text = []
        else:
            accumulated_text.append(char_or_braces)

        previous_char_or_braces = char_or_braces

    append_to_parts("".join(accumulated_text))

    parsed_command: ParsedCommand = ParsedCommand()
    parsed_command.parts = parts
    return parsed_command


def verify_command(command: ParsedCommand) -> None:
    assert command.parts, "Empty command"

    assert len(command.parts) >= 1, "Command should have at least 1 part!"

    if len(command.parts) == 1:
        assert (
            command.parts[0] in KEYWORDS_WITH_0_ARGS
        ), f"Command {command.parts[0]} should have an argument!"

    for part1, part2 in zip(command.parts, command.parts[1:]):
        if part1 in KEYWORDS_WITH_STR_ARG:
            assert isinstance(part2, str), f"Command {part1} should have an argument!"
            assert part2, "No empty arguments!"

        if part1 is LanguageKeywords.SEEK:
            assert isinstance(part2, str) or isinstance(part2, SEEK_ARGS)
            _ = parse_SEEK_arg(part2)

        if part1 in KEYWORDS_WITH_0_ARGS:
            assert isinstance(
                part2, LanguageKeywords
            ), f"Command {part1} should have no arguments!"

        if isinstance(part2, str):
            assert isinstance(
                part1, LanguageKeywords
            ), f"Argument {part2} is not for any command!"

            assert (
                part1 in KEYWORDS_WITH_STR_ARG or part1 is LanguageKeywords.SEEK
            ), f"Argument {part2} is not for a suitable command, instead {part1}!"

        if isinstance(part2, SEEK_ARGS):
            assert part1 is LanguageKeywords.SEEK


class GameState:
    def __init__(self, file: Path) -> None:
        print(f"GameState init, file: {file}")
        with file.open("r") as f_:
            self.program: str = f_.read()

        self.current_field: str = remove_newlines_inside_commands(self.program)

        self.beginning_cursor: int = 0
        self.ending_cursor: int = 0

        self.state: StatesOfGame = StatesOfGame.FINDING_COMMAND

        self.current_command: Optional[ParsedCommand] = None
        self.current_command_beg_and_text: Optional[Tuple[int, str]] = None

        self.current_command_part_idx: int = 0

        self.execution_report: List[str] = ["Not run yet"]

        self.finished_running_for_automatic_processing: bool = False

    def set_ERROR_state(self, error_report: str) -> None:
        self.state = StatesOfGame.ERROR
        self.execution_report.append(error_report)

    def execute_state(self) -> None:
        self.execution_report = []

        if self.state == StatesOfGame.FINDING_COMMAND:
            self.execute_FINDING_COMMAND()
        elif self.state == StatesOfGame.EXECUTING_COMMAND_PART:
            self.get_and_execute_command_part()
        elif self.state == StatesOfGame.DELETING_COMMAND:
            self.execute_DELETING_COMMAND()
        elif self.state == StatesOfGame.FINISHED:
            self.execution_report.append("Program execution finished!")
            self.finished_running_for_automatic_processing = True
        else:
            self.execution_report.append(
                "Error encountered! No further execution is possible!"
            )

    def execute_FINDING_COMMAND(self) -> None:
        try:
            part_before_braces, command_text, part_after_braces = (
                get_first_encountered_braces(self.current_field)
            )
            if not command_text:
                self.state = StatesOfGame.FINISHED
                self.execution_report.append("No commands found, finished!")
                return

            command_unpacked = unpack_command(command_text)
            command_parsed = parse_unpacked(command_unpacked)
            verify_command(command_parsed)
            self.current_command = command_parsed
        except StopIteration:
            self.state = StatesOfGame.FINISHED
            self.execution_report.append("Execution finished!")
            return
        except AssertionError as e:
            self.set_ERROR_state(str(e.args))
            return
        except Exception as e:
            self.set_ERROR_state(f"Error! {e}")
            return

        self.beginning_cursor, self.ending_cursor = (
            len(part_before_braces),
            len(self.current_field) - len(part_after_braces),
        )
        self.current_command_beg_and_text = (self.beginning_cursor, command_text)

        self.execution_report.append(f"""Found command: '{self.current_command}'""")

        self.state = StatesOfGame.EXECUTING_COMMAND_PART

    def execute_DELETING_COMMAND(self) -> None:
        assert self.current_command_beg_and_text
        first_part_of_field: str = self.current_field[
            : self.current_command_beg_and_text[0]
        ]
        last_part_of_field: str = self.current_field[
            self.current_command_beg_and_text[0]
            + len(self.current_command_beg_and_text[1]) :
        ]
        self.current_field = first_part_of_field + last_part_of_field
        self.current_command_beg_and_text = None
        self.current_command_part_idx = 0
        self.beginning_cursor = 0
        self.ending_cursor = 0

        self.execution_report.append("Deleted a command")
        self.state = StatesOfGame.FINDING_COMMAND

    def get_and_execute_command_part(self) -> None:
        assert self.current_command

        current_command_part = self.current_command.parts[self.current_command_part_idx]
        assert isinstance(current_command_part, LanguageKeywords)

        command_argument: Union[str, SEEK_ARGS, None] = None
        if current_command_part not in KEYWORDS_WITH_0_ARGS:
            arg = self.current_command.parts[self.current_command_part_idx + 1]
            if current_command_part is LanguageKeywords.SEEK:
                assert isinstance(arg, str) or isinstance(arg, SEEK_ARGS)
                _ = parse_SEEK_arg(arg)
            else:
                assert isinstance(arg, str)

            command_argument = arg

        self.execute_keyword(current_command_part, command_argument)

    def execute_keyword(
        self, command: LanguageKeywords, argument: Union[None, SEEK_ARGS, str]
    ) -> None:
        success: bool = self.select_and_execute_keyword_action(command, argument)
        self.execution_report.append(f"Executing {command}, success: {success}")

        if success:
            self.advance_execution(command, argument)
        else:
            self.state = StatesOfGame.DELETING_COMMAND

    def select_and_execute_keyword_action(
        self, command: LanguageKeywords, argument: Union[None, SEEK_ARGS, str]
    ) -> bool:
        if command is LanguageKeywords.FIND:
            assert isinstance(argument, str)
            return self.execute_FIND(argument)
        elif command is LanguageKeywords.REMOVE:
            assert isinstance(argument, str)
            return self.execute_REMOVE(argument)
        elif command is LanguageKeywords.DELETE:
            assert not argument
            return self.execute_DELETE()
        elif command is LanguageKeywords.REPLACE:
            assert isinstance(argument, str)
            return self.execute_REPLACE(argument)
        elif command is LanguageKeywords.SEEK:
            assert argument is not None
            return self.execute_SEEK(argument)
        elif command is LanguageKeywords.INSERT:
            assert isinstance(argument, str)
            return self.execute_INSERT(argument)
        elif command is LanguageKeywords.APPEND:
            assert isinstance(argument, str)
            return self.execute_APPEND(argument)
        elif command is LanguageKeywords.COMMENT:
            return False
        elif command is LanguageKeywords.ONCE:
            return False
        else:
            raise NotImplementedError()

    def try_relocate_start_of_command_on_removal(
        self, len_of_part_before_removed: int, len_of_removed_part: int
    ) -> None:
        assert self.current_command_beg_and_text is not None
        if (
            self.current_command_beg_and_text[0]
            + len(self.current_command_beg_and_text[1])
            > len_of_part_before_removed
        ):
            assert (
                self.current_command_beg_and_text[0]
                >= len_of_part_before_removed + len_of_removed_part
            )
            tup = self.current_command_beg_and_text
            tup = (tup[0] - len_of_removed_part, tup[1])
            self.current_command_beg_and_text = tup

    def try_relocate_start_of_command_on_insertion(
        self, idx_of_insertion: int, len_of_insertion: int
    ) -> None:
        assert self.current_command_beg_and_text
        if (
            self.current_command_beg_and_text[0]
            + len(self.current_command_beg_and_text[1])
            > idx_of_insertion
        ):
            tup = self.current_command_beg_and_text
            tup = (tup[0] + len_of_insertion, tup[1])
            self.current_command_beg_and_text = tup

    def execute_FIND(self, argument: str) -> bool:
        try:
            argument = unescape_those_unprotected_by_braces(argument)

            result: int = find_in_text_with_protection_from_braces(
                self.current_field,
                argument,
                self.ending_cursor,
                len(self.current_field),
            )

            self.beginning_cursor = result
            self.ending_cursor = self.beginning_cursor + len(argument)
            return True
        except ValueError:
            return False

    def execute_REMOVE(self, argument: str) -> bool:
        try:
            argument = unescape_those_unprotected_by_braces(argument)

            result: int = find_in_text_with_protection_from_braces(
                self.current_field, argument, self.beginning_cursor, self.ending_cursor
            )

            part_before: str = self.current_field[:result]
            part_after: str = self.current_field[result + len(argument) :]

            self.try_relocate_start_of_command_on_removal(
                len(part_before), len(argument)
            )

            self.current_field = part_before + part_after

            self.beginning_cursor = self.ending_cursor = len(part_before)

            return True
        except ValueError:
            return False

    def execute_DELETE(self) -> bool:
        part_before: str = self.current_field[: self.beginning_cursor]
        part_after: str = self.current_field[self.ending_cursor :]

        self.try_relocate_start_of_command_on_removal(
            len(part_before), self.ending_cursor - self.beginning_cursor
        )

        self.current_field = part_before + part_after

        self.beginning_cursor = self.ending_cursor = len(part_before)

        return True

    def execute_REPLACE(self, argument: str) -> bool:
        argument = unescape_those_unprotected_by_braces(argument)
        if argument.strip() == "SELF":
            assert self.current_command_beg_and_text is not None
            argument = self.current_command_beg_and_text[1]

        try:
            part_before: str = self.current_field[: self.beginning_cursor]
            part_after: str = self.current_field[self.ending_cursor :]

            self.try_relocate_start_of_command_on_removal(
                len(part_before), self.ending_cursor - self.beginning_cursor
            )
            self.try_relocate_start_of_command_on_insertion(
                len(part_before), len(argument)
            )

            self.current_field = part_before + argument + part_after

            self.ending_cursor = self.beginning_cursor + len(argument)

            return True
        except ValueError:
            return False

    def execute_SEEK(self, argument: Union[str, SEEK_ARGS]) -> bool:
        to_set: int
        if argument is SEEK_ARGS.SOL:
            to_set = (
                find_char_backwards(self.current_field, self.beginning_cursor - 1, "\n")
                + 1
            )
        elif argument is SEEK_ARGS.EOL:
            try:
                to_set = self.current_field.index("\n", self.ending_cursor)
            except:
                to_set = len(self.current_field)
        elif argument is SEEK_ARGS.SOF:
            to_set = 0
        elif argument is SEEK_ARGS.EOF:
            to_set = len(self.current_field)
        else:
            assert isinstance(argument, str)

            to_int: int = int(argument)
            if to_int > 0:
                to_set = self.ending_cursor + to_int
            elif to_int < 0:
                to_set = self.beginning_cursor + to_int
            else:
                return True

        to_set = max(0, to_set)
        to_set = min(len(self.current_field), to_set)

        self.beginning_cursor = to_set
        self.ending_cursor = to_set

        return True

    def execute_INSERT(self, argument: str) -> bool:
        argument = unescape_those_unprotected_by_braces(argument)
        if argument.strip() == "SELF":
            assert self.current_command_beg_and_text is not None
            argument = self.current_command_beg_and_text[1]

        first_part: str = self.current_field[: self.beginning_cursor]
        last_part: str = self.current_field[self.beginning_cursor :]

        self.try_relocate_start_of_command_on_insertion(len(first_part), len(argument))

        self.current_field = first_part + argument + last_part

        self.beginning_cursor += len(argument)
        self.ending_cursor += len(argument)
        return True

    def execute_APPEND(self, argument: str) -> bool:
        argument = unescape_those_unprotected_by_braces(argument)
        if argument.strip() == "SELF":
            assert self.current_command_beg_and_text is not None
            argument = self.current_command_beg_and_text[1]

        first_part: str = self.current_field[: self.ending_cursor]
        last_part: str = self.current_field[self.ending_cursor :]

        self.try_relocate_start_of_command_on_insertion(len(first_part), len(argument))

        self.current_field = first_part + argument + last_part
        return True

    def advance_execution(
        self, command: LanguageKeywords, argument: Union[None, SEEK_ARGS, str]
    ) -> None:
        self.current_command_part_idx += 1

        if argument is not None:
            self.current_command_part_idx += 1

        assert self.current_command
        if self.current_command_part_idx >= len(self.current_command.parts):
            self.state = StatesOfGame.FINDING_COMMAND
            self.current_command_part_idx = 0
            self.execution_report.append(
                f"Command {command} executed, was the last one in the list!"
            )
            self.beginning_cursor = 0
            self.ending_cursor = 0
            self.current_command = None
            self.current_command_beg_and_text = None

    def print_execution_report(self) -> None:
        for line in self.execution_report:
            print(line)

    def process_output_command(self, command: str) -> None:
        try:
            split = command.split(" ")
            assert split[0] == "output", "Didn't understand this command!"
            assert len(split) > 1
            with Path(split[1]).open("w") as file:
                file.write(self.current_field)
                print(f"Written current field state to {split[1]}!")
        except Exception as e:
            print(f"Trying to write state of current field, error: {e}")

    def print_snapshot(self) -> None:
        field: str = self.current_field
        beg, end = self.beginning_cursor, self.ending_cursor
        if beg == end:
            part_before = field[:beg]
            part_after = field[beg:]
            print(part_before + "‹" + part_after)
        else:
            part_before = field[:beg]
            part_after = field[end:]
            part_inside = field[beg:end]
            print(part_before + "‹" + part_inside + "›" + part_after)


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Toy esolang")
    parser.add_argument(
        "--run",
        action="store",
        help="Run a script",
        required=True,
        nargs=1,
        type=Path,
        metavar="SCRIPT_PATH",
    )
    args = parser.parse_args()

    if args.run:
        start_running(args.run[0])
    else:
        print("Use --run to begin the game")
        sys.exit(1)


def start_running(file: Path) -> None:
    print(f"running file {file}")
    event_loop(file)


def event_loop(file: Path) -> None:
    GAME_STATE: GameState = GameState(file)

    while True:
        command = (
            input("(press Enter or type 'help' and press Enter) > ").strip().lower()
        )

        if (
            command == "stop"
            or command == "quit"
            or command == "exit"
            or command == "e"
            or command == "q"
        ):
            print("Thanks for playing around!")
            break
        elif command == "":
            GAME_STATE.execute_state()
            GAME_STATE.print_execution_report()
        elif command == "snap":
            GAME_STATE.print_snapshot()
        elif command == "help":
            print(
                """
The Newspeak interpreter accepts the following commands:
  simply pressing Enter without entering any command will advance 1 turn of processing the current program and output the report on processing. The vast majority of progams need several steps to complete!
  help will output this message.
  stop, exit, quit, e, q are all aliases of each other. They stop the current interpeter session.
  snap will output the current state of the field being processed.
  output [file name with extension] will output the current state of the board to a file at the passed argument.
  process will enter the intepreter into a processing mode. In this mode, processing turns will advance without the input from the editor. To stop it, enter s (as in stop) and press Enter.
"""
            )
        elif command == "process":
            GAME_STATE.finished_running_for_automatic_processing = False
            processing_mode(GAME_STATE)
        elif command.find("output") >= 0:
            GAME_STATE.process_output_command(command)
        else:
            print("I don't understand that command.")


def processing_mode(game_state: GameState) -> None:
    def process_turns():
        while not game_state.finished_running_for_automatic_processing:
            print("Processing turn (press 's' then Enter to stop)...")
            game_state.execute_state()
            game_state.print_execution_report()
            time.sleep(0.1)

    processing_thread = threading.Thread(target=process_turns)
    processing_thread.start()

    while True:
        command = (
            input("Processing mode (press 's' then Enter to stop) > ").strip().lower()
        )
        if command == "s":
            game_state.finished_running_for_automatic_processing = True
            processing_thread.join()
            print("Exiting processing mode.")
            break


if __name__ == "__main__":
    main()
