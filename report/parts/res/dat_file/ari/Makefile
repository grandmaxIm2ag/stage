CC= python
SCRIPT=script_plot.py 
OLD_SCRIPT=script_plot_old.py
OLD_SCRIPT2=script_plot_old2.py

SRC=$(wildcard *.dat)
OBJ=$(SRC:.dat=.png)
SRC_old=$(wildcard *.dat_old)
OBJ_old=$(SRC_old:.dat_old=.jpeg)
SRC_old2=$(wildcard *.dat_old2)
OBJ_old2=$(SRC_old2:.dat_old2=.jpg)

TARGET=main main_old main_old2

all:${TARGET}

main: ${OBJ}

main_old: ${OBJ_old}

main_old2: ${OBJ_old2}

%.png: %.dat
	${CC} ${SCRIPT} $^ $@

%.jpeg: %.dat_old
	${CC} ${OLD_SCRIPT} $^ $@

%.jpg: %.dat_old2
	${CC} ${OLD_SCRIPT2} $^ $@

clean:
	rm -rf ${OBJ_old2} ${OBJ_old} ${OBJ} ${TARGET} *~
