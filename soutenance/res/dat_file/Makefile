CC= python
SCRIPT=script_plot.py

SRC=$(wildcard *.dat)
OBJ=$(SRC:.dat=.png)

TARGET=main

all:${TARGET}

main: ${OBJ}
	cd ari ; \
	make all
	cd acc ; \
	make all
	cd nmi ; \
	make all
	cd trash_class ; \
	make all
clean:
	rm -rf ${OBJ} ${TARGET} *~
