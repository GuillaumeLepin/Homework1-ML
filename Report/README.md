# Report (LaTeX)

This folder contains the LaTeX report skeleton for Homework 1.

## Structure

- `main.tex`: report entry point
- `sections/`: one file per homework question
- `figures/`: local report figures (optional)
- `bibliography/references.bib`: bibliography placeholder
- `Makefile`: quick compile/clean commands

## Compile

From this `Report/` directory:

```bash
make pdf
```

or directly:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

## Clean build artifacts

```bash
make clean
```
