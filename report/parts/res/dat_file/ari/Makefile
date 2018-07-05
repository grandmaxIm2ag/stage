CC= python
SCRIPT=script_plot.py

SRC=$(wildcard *.dat)
OBJ=$(SRC:.dat=.png)

TARGET=main

all:${TARGET}

main: ${OBJ}

%.png: %.dat
	${CC} ${SCRIPT} $^ $@

clean:
	rm -rf ${OBJ} ${TARGET} *~
