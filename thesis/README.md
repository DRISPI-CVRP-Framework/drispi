# Thesis LaTeX project

TUM-style academic thesis template (KOMA-Script `scrreprt`), built with **pdflatex**. Entry point: `main.tex` → output: `main.pdf`.

---

## Quick rebuild (terminal)

From this folder:

```bash
cd thesis
latexmk -pdf main.tex
```

That runs pdflatex, bibtex, and glossaries as configured in `.latexmkrc`.

| Command | Purpose |
|---|---|
| `latexmk -pdf main.tex` | Build / rebuild `main.pdf` |
| `latexmk -C` | Clean generated aux files |
| `latexmk -C && latexmk -pdf main.tex` | Clean rebuild |
| `latexmk -pdf -pvc main.tex` | Watch mode: rebuild on save |

Manual chain (only if you skip latexmk):

```bash
pdflatex main.tex
bibtex main
makeglossaries main
pdflatex main.tex
pdflatex main.tex
```

---

## Build through the UI (Cursor / VS Code)

Requires the **LaTeX Workshop** extension (`James-Yu.latex-workshop`).

1. Open any `.tex` file under `thesis/` (preferably a chapter, or `main.tex`).
2. Magic comments in `main.tex` tell the extension what to use:
   - `% !TEX program = pdflatex`
   - `% !TEX root = main.tex`  
   So editing a chapter still builds the whole thesis from `main.tex`.
3. **Build**
   - Command Palette → `LaTeX Workshop: Build LaTeX project`, or
   - green ▶ / TeX build button in the editor toolbar, or
   - default shortcut: `Ctrl+Alt+B` (Linux/Windows) / `Cmd+Option+B` (macOS).
4. **View PDF**
   - Command Palette → `LaTeX Workshop: View LaTeX PDF`, or
   - `Ctrl+Alt+V` / click the PDF icon.
   - Side-by-side preview is the usual setup; SyncTeX jump works if enabled (click in PDF ↔ source).
5. **Auto-build on save** is usually on by default in LaTeX Workshop (`latex-workshop.latex.autoBuild.run`). Save a chapter → PDF updates.

If build fails in the UI, open the LaTeX Workshop output / problem panel, or check `main.log` in this folder.

---

## What each file / folder does

### Core

| Path | Role |
|---|---|
| `main.tex` | Root document: class options, include order, front/main/back matter |
| `settings.tex` | Packages, margins, fonts, headers, citations, glossaries, hyperref |
| `commands.tex` | Custom macros and theorem-like environments |
| `info.tex` | Cover metadata (title, authors, examiner, date, …) — **edit this for personal data** |
| `cover.tex` | Cover page layout (reads macros from `info.tex`; usually leave alone) |
| `.latexmkrc` | latexmk recipe: pdflatex + bibtex + makeglossaries |

### Content

| Path | Role |
|---|---|
| `chapters/0_Abstract.tex` | Abstract |
| `chapters/1_Introduction.tex` … `6_Conclusion.tex` | Main chapters |
| `chapters/Appendix.tex` | Appendix |
| `chapters/Declaration of Authorship.tex` | Authorship declaration |
| `chapters/Declaration of AI Use.tex` | AI-use declaration |
| `Abbr.tex` | Acronym / abbreviation glossary entries |
| `Symbols.tex` | Symbol glossary entries |

### Bibliography & assets

| Path | Role |
|---|---|
| `bibliography/literature.bib` | BibTeX references |
| `bibliography/biblio_style.bst` | Citation / bibliography style |
| `images/` | Figures (e.g. EPS/PDF/PNG used by `\includegraphics`) |

### Generated (do not edit)

`main.pdf`, `main.aux`, `main.log`, `main.toc`, `main.lof`, `main.lot`, glossary intermediates (`.acr`, `.syi`, …), etc. Safe to delete via `latexmk -C` and rebuild.

---

## Document flow (`main.tex`)

1. **Cover** (`cover.tex`)
2. **Front matter** (roman page numbers): abstract → TOC → lists of figures/tables → abbreviations & symbols
3. **Main matter** (arabic from Introduction): chapters 1–6
4. **Bibliography** (`literature.bib`)
5. **Back matter**: appendix, declarations

---

## Main settings (summary)

Defined in `main.tex` + `settings.tex`:

- Class: `scrreprt`, A4, **12 pt**
- Margins: left **50 mm**, right 20 mm, top 20 mm, bottom 10 mm
- Language: English; font: Times; body: 1.5 line spacing
- Citations: `natbib` (round) + custom BST
- Glossaries: acronyms + symbols
- TOC / section numbering depth: 4

To change layout, packages, or citation style → edit `settings.tex`.  
To change title/authors/examiner → edit `info.tex`.  
To add a chapter → create `chapters/….tex` and `\input{…}` it in `main.tex`.
