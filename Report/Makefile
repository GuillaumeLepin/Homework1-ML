LATEXMK=latexmk
MAIN=main.tex

.PHONY: all pdf clean

all: pdf

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode $(MAIN)

clean:
	$(LATEXMK) -c
	rm -f *.bbl *.blg *.run.xml
