## Newspeak is a self-modifying substitution-based language for text editing (esolang). Newspeak is employed by the Ministry of Truth (MINITRUE) for live editing of news, truth, and true news. Never speak falsity with Newspeak!

To run Newspeak, launch with python newspeak.py with argument --run program.txt. This will launch the file program.txt in interactive mode.

`python newspeak.py --run program.txt`

Newspeak was created with python 3.11.8, the further from this version the less likely it will run.

-----

Newspeak operates on lists of commands, with some commands accepting an argument. Here is an example of a command list:

`{FIND|deceased|REPLACE|born}`

Most programs in Newspeak consist of several lists of commands:

```
{FIND|Jhonson|INSERT|comrade }
{FIND|Stevenson|INSERT|traitor }
```

Lists of commands can contain other lists:

`{FIND|@|REPLACE|{FIND|#|REPLACE|@}}`

On execution, the above command will replace the mark "@" with the following command:

`{FIND|#|REPLACE|@}`

When executing a file, Newspeak will search for the first valid command list it encounters. After finding one, it executes every command one after another. If any of them fail, the command list is deleted from the file. If every command in a list executes successfully, Newspeak resets itself to start executing from the beginning.

This ensures that if a command is able to be executed, it is executed on every part of the file. However, it is a job of an editor at MINITRUE to keep programs from going into infinite loops.

Because lists and text they edit coexist in the same space, Newspeak can be said to be self-modifying. 

------

Full list of commands for the interactive Newspeak interpreter:
 - simply pressing Enter without entering any command will advance 1 turn of processing the current program and output the report on processing. The vast majority of progams need several steps to complete!
 - help will output this message.
 - stop, exit, quit, e, q are all aliases of each other. They stop the current interpeter session.
 - snap will output the current state of the board being processed.
 - output [file name with extension] will output the current state of the board to a file at the passed argument.
 - process will enter the intepreter into a processing mode. In this mode, processing turns will advance without the input from the editor. To stop it, enter s (as in stop) and press Enter.

------

Full list of Newspeak language keywords:
 - FIND|arg: finds the argument in text, starting from after the end of the current command. If finds, sets starting and ending cursors to the edges of the argument. If doesn't find, fails the current command list.
 - REMOVE|arg: finds the argument between the current starting and ending cursors. If finds, deletes this text from the field. If doesn't find, fails the current command list.
 - DELETE: Has no argument. Simply deletes everything between current starting and ending cursors. Always succeeds.
 - REPLACE|arg: replaces everything between current starting and ending cursors with the argument. Always succeeds. Can accept SELF.
 - INSERT|arg: inserts the argument at the current starting cursor. Always succeeds. Can accept SELF.
 - APPEND|arg: inserts the argument at the current ending cursor. Always succeeds. Can accept SELF.
 - ONCE: does nothing. Accepts no arguments. Always fails. Useful for making commands one-time-use-only.
 - SEEK|arg: moves the starting and ending cursors to coincide according to arg. Arg can take one of the following forms:
 - - SOL: start of line, relative to current beginning cursor.
   - EOL: end of line, relative to current ending cursor.
   - SOF: start of file.
   - EOF: end of file.
   - n: can be any integer.
   - - If positive, moves the ending cursor by n characters right and beginning cursor to coincide with it.
     - If negative, moves the starting cursor by n characters left and ending cursor to coincide with it.
     - If 0, does nothing.
 - Finally, arguments to REPLACE, INSERT and APPEND can be SELF. In this case, the text that is inserted with these commands is the current command's very text.

-------

Additional features of the language include:
 - Comments can be included in the program with \<angle brackets\>. They need to be paired and can't be nested for now. Comments are removed from the file before running it.
 - Commands can include \\n newlines and \\t tabs for better readability. These characters are removed from commands before running the file.

