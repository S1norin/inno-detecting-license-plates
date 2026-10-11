---
name: read-pdf
description: Extract text, markdown, tables, and images from PDF files using the liteparse CLI (`lit`). Use for reading PDFs, extracting syllabi/sections, converting PDFs to markdown, or getting structured JSON with bounding boxes.
---

# Read PDF (liteparse)

Parse PDFs locally with the `lit` CLI from
[liteparse](https://github.com/run-llama/liteparse) (Rust core, fast, no cloud).
Installed globally via `npm i -g @llamaindex/liteparse`.

Verify: `lit --version`

## Basic usage

```bash
# Markdown (best for reading/LLM consumption): headings, tables, lists, links
lit parse document.pdf --format markdown -o output.md

# Plain text with layout preserved (default format)
lit parse document.pdf -o output.txt

# Structured JSON with bounding boxes
lit parse document.pdf --format json -o output.json
```

Print to stdout by omitting `-o` (useful for short docs):

```bash
lit parse document.pdf --format markdown
```

## Targeting pages

Useful for extracting a section (e.g., a syllabus from intro slides) without
parsing the whole document:

```bash
lit parse document.pdf --format markdown --target-pages "1-12"
lit parse document.pdf --format markdown --target-pages "1-5,10,15-20"
lit parse document.pdf --format markdown --max-pages 100
```

## OCR

OCR (Tesseract, bundled) is on by default — needed for scanned/image-heavy PDFs
but slows parsing. Disable for native-text PDFs:

```bash
lit parse document.pdf --format markdown --no-ocr
lit parse document.pdf --format markdown --ocr-language "eng+rus"
```

## Images and links (markdown mode)

```bash
# Drop image placeholders (cleaner text)
lit parse document.pdf --format markdown --image-mode off

# Extract embedded images to disk and reference them
lit parse document.pdf --format markdown --image-mode embed \
  --extract-images --image-output-dir ./images

# Links as plain text
lit parse document.pdf --format markdown --no-links
```

## JSON extras

```bash
lit parse document.pdf --format json --extract-vector-graphics
lit parse document.pdf --format json --extract-text-metadata
lit parse document.pdf --format json --extract-annotations
lit parse document.pdf --format json --extract-form-fields
lit parse document.pdf --format json --extract-blocks   # layout blocks with bboxes
```

## Remote PDFs

```bash
curl -sL https://example.com/report.pdf | lit parse - --format markdown
```

## Tips

- Lecture slide PDFs: markdown mode reconstructs headings well; two-column
  slide layouts can interleave columns — read output critically.
- OCR on (default) is ~2-20s per page for image-heavy slides; `--no-ocr` is
  much faster when the PDF has a text layer.
- For very large docs, parse page ranges in chunks and save each to a file.
