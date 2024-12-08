Newspeak is a self-modifying substitution-based language for text editing (esolang). Newspeak is employed by the Ministry of Truth (MINITRUE) for live editing of news, truth, and true news. Never speak falsity with Newspeak!

Newspeak operates on lists of commands, with some commands accepting an argument. Here is an example of a command list:

{FIND|deceased|REPLACE|born}

Most programs in Newspeak consist of several lists of commands:

{FIND|Jhonson|INSERT|comrade }
{FIND|Stevenson|INSERT|traitor }

Lists of commands can contain other lists:

{FIND|@|REPLACE|{FIND|#|REPLACE|@}}

On execution, the above command will replace the mark "@" with the following command:

{FIND|#|REPLACE|@}

When executing a file, Newspeak will search for the first valid command list it encounters. After finding one, it executes every command one after another. If any of them fail, the command list is deleted from the file. If every command in a list executes successfully, Newspeak resets itself to start executing from the beginning.

This ensures that if a command is able to be executed, it is executed on every part of the file. However, it is a job of an editor at MINITRUE to keep programs from going into infinite loops.

Because lists and text they edit coexist in the same space, Newspeak can be said to be self-modifying. 

To run Newspeak, launch with python newspeak.py with argument --start program.txt. This will launch the file program.txt in interactive mode.

Newspeak was created with python 3.11.8, the further from this version the less likely it will run.
