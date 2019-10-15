SRC_Expose=template
SRC_THESIS=Thesis/main.tex
# Figure out which machine we're running on.
UNAME=$(shell uname -s)
DIR=$(shell pwd)


all: poster

B:
	latexmk -pdf $(SRC_THESIS)
	#pdflatex $(SRC_THESIS)
	#bibtex8 -B $(SRC_THESIS) || if [ $$? -ne 1 ] ; then $$?; 

	# pdflatex $(SRC_THESIS)
	# pdflatex $(SRC_THESIS)
	# pdflatex $(SRC_THESIS)

expose:
	pdflatex $(SRC_Expose)
	bibtex8 -B $(SRC_Expose) || if [ $$? -ne 1 ] ; then $$?; fi

	pdflatex $(SRC_Expose)
	pdflatex $(SRC_Expose)
	pdflatex $(SRC_Expose)
	rm expose.pdf
	mv template.pdf expose.pdf

thesis:
	pdflatex $(SRC_THESIS)
	bibtex8 -B $(SRC_THESIS) || if [ $$? -ne 1 ] ; then $$?; fi

	pdflatex $(SRC_THESIS)
	pdflatex $(SRC_THESIS)
	pdflatex $(SRC_THESIS)
	
wo-java:
	pdflatex $(SRC_Expose)
	bibtex8 $(SRC_Expose) || if [ $$? -ne 1 ] ; then $$?; fi
	pdflatex $(SRC_Expose)
	pdflatex $(SRC_Expose)
	pdflatex $(SRC_Expose)

	pdflatex $(SRC_THESIS)
	bibtex8 $(SRC_THESIS) || if [ $$? -ne 1 ] ; then $$?; fi
	pdflatex $(SRC_THESIS)
	pdflatex $(SRC_THESIS)
	pdflatex $(SRC_THESIS)

clean:
	rm -f *.aux *.lof *.log *.lot *.fls *.out *.toc *.fmt *.idx *.ilg *.ind *.ist *.bbl *.cut *.nav *.snm
	rm -f *.bcf *.blg *-blx.aux *-blx.bib *.run.xml *.upa *.upb *.fdb_latexmk *.synctex
	rm -f *.synctex\(busy\) *.synctex.gz *.synctex.gz\(busy\) *.pdfsync

clean-deep:
	find . -regextype posix-extended -regex ".*(\.(aux|lof|log|lot|fls|out|toc|fmt|idx|cut))" -exec rm {} +
	find . -regextype posix-extended -regex ".*(\.(ilg|ind|ist|bbl|bcf|blg|upa|upb))" -exec rm {} +
	find . -regextype posix-extended -regex ".*(\.(synctex|synctex\(busy\)|synctex\.gz|synctex\.gz\(busy\)))" -exec rm {} +
	find . -regextype posix-extended -regex ".*(\-blx\.bib|\.(\-blx\.aux|run\.xml|fdb\_latexmk|pdfsync))" -exec rm {} +

.PHONY: clean poster
