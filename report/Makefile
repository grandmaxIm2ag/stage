CC=pdflatex
BIB=biber

RES_REP=parts/res/dat_file

SRC=$(wildcard *.tex)
OBJ=$(SRC:.tex=)
all: pdf

pdf: ${SRC} res
	${CC} ${OBJ}
	biber ${OBJ}
	latex_count=8 ; \
	while egrep -s 'Rerun (LaTeX|to get cross-references right)' ${OBJ}.log && [ $$latex_count -gt 0 ] ;\
	    do \
	      echo "Rerunning latex...." ;\
	      ${CC} ${OBJ} ;\
	      latex_count=`expr $$latex_count - 1` ;\
	    done

res:
	cd ${RES_REP} ; \
	make all

clean_res:
	cd ${RES_REP} ; \
	make clean

clean: clean_res
	rm -f *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *.brf *.blg *.bbl ${OBJ} *.bcf *.fls *.run.xml *.fdb_latexmk *~
