import codecs
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from math import hypot
from pathlib import Path
from zipfile import ZipFile


@dataclass(frozen=True)
class Block:
    text: str
    location: str = ""
    kind: str = "text"
    level: int = 0


def decode(data):
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("gb18030")


def table_text(rows):
    return "\n".join(" | ".join(str(value).replace("\n", " ") if value is not None else ""
                                 for value in row) for row in rows)


def pdf_blocks(data):
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        if pdf.needs_pass or len(pdf) > 500:
            raise ValueError("PDF 已加密或超过 500 页。")
        pages = []
        for page in pdf:
            lines = page.get_text(sort=True).splitlines()
            text_chars = sum(ch.isalnum() for line in lines for ch in line)
            images = page.get_image_info() if text_chars < 40 else []
            raster_page = any(abs(pymupdf.Rect(image["bbox"])) >= abs(page.rect) * .5 for image in images)
            if images and (not text_chars or raster_page):
                tessdata = os.getenv("TESSDATA_PREFIX")
                language = os.getenv("INKATLAS_OCR_LANG", "chi_sim+eng")
                try:
                    if tessdata and any(not (Path(tessdata) / f"{lang}.traineddata").is_file()
                                        for lang in language.split("+")):
                        raise ValueError("Tesseract 缺少指定语言的数据。")
                    ocr = page.get_textpage_ocr(
                        language=language, dpi=200, full=True, tessdata=tessdata,
                    )
                except (RuntimeError, ValueError) as error:
                    raise ValueError(f"PDF 第 {page.number + 1} 页需要 OCR，请检查 Tesseract 语言数据。") from error
                lines = page.get_text(textpage=ocr, sort=True).splitlines()
            tables = [table_text(table.extract()) for table in page.find_tables().tables]
            watermark = set()
            for item in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
                for line in item.get("lines", []):
                    for span in line["spans"]:
                        if span.get("alpha", 255) < 100 and span["size"] >= 18:
                            watermark.add(span["text"].strip())
            lines = [line.strip() for line in lines if line.strip() and line.strip() not in watermark]
            pages.append((lines, tables))
    edges = Counter(line for lines, _ in pages if len(lines) >= 3 for line in set((lines[0], lines[-1])))
    for index, (lines, tables) in enumerate(pages, 1):
        if len(lines) >= 3:
            lines = [line for i, line in enumerate(lines)
                     if i not in (0, len(lines) - 1)
                     or (edges[line] < 2 and not re.fullmatch(r"(?:第\s*)?\d+(?:\s*页)?", line))]
        yield Block("\n".join(lines), f"第 {index} 页")
        yield from (Block(table, f"第 {index} 页 · 表 {i}", "table") for i, table in enumerate(tables, 1))


def office_blocks(extension, data):
    with ZipFile(BytesIO(data)) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 100_000_000:
            raise ValueError("Office 文件解包后超过 100 MB。")
    if extension == ".docx":
        from docx import Document
        from docx.table import Table

        for index, item in enumerate(Document(BytesIO(data)).iter_inner_content(), 1):
            if isinstance(item, Table):
                yield Block(table_text([cell.text for cell in row.cells] for row in item.rows),
                            f"元素 {index}", "table")
            else:
                style = item.style.name
                match = re.fullmatch(r"Heading (\d+)", style)
                level = int(match[1]) + 1 if match else int(style == "Title")
                yield Block(item.text, f"段落 {index}", "heading" if level else "text", level)
    elif extension == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(BytesIO(data), read_only=True, data_only=True, keep_links=False)
        try:
            for sheet in workbook.worksheets:
                if (sheet.max_row or 1) * (sheet.max_column or 1) > 250_000:
                    raise ValueError(f"工作表 {sheet.title} 超过 25 万单元格。")
                for index, row in enumerate(sheet.iter_rows(values_only=True), 1):
                    if any(value is not None for value in row):
                        yield Block(table_text([row]), f"{sheet.title} · 第 {index} 行", "table")
        finally:
            workbook.close()
    else:
        from pptx import Presentation

        for index, slide in enumerate(Presentation(BytesIO(data)).slides, 1):
            for shape in slide.shapes:
                location = f"第 {index} 张幻灯片"
                if shape.has_table:
                    yield Block(table_text([cell.text for cell in row.cells] for row in shape.table.rows),
                                location, "table")
                elif shape.has_text_frame:
                    title = shape == slide.shapes.title
                    yield Block(shape.text, location, "heading" if title else "text", int(title))


def html_blocks(data):
    from bs4 import BeautifulSoup

    html = BeautifulSoup(data, "html.parser")
    for tag in html.select("script, style, noscript, nav, footer, header, [hidden]"):
        tag.decompose()
    root = html.find("main") or html.body or html
    elements = root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"])
    for index, tag in enumerate(elements, 1):
        if tag.find_parent("table"):
            continue
        if tag.name == "table":
            rows = ([cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                    for row in tag.find_all("tr"))
            yield Block(table_text(rows), f"HTML 元素 {index}", "table")
        else:
            level = int(tag.name[1]) if re.fullmatch(r"h[1-6]", tag.name) else 0
            yield Block(tag.get_text(" ", strip=True), f"HTML 元素 {index}", "heading" if level else "text", level)
    if not elements:
        yield Block(root.get_text("\n", strip=True))


def parse_file(name, data):
    extension = Path(name).suffix.lower()
    if extension == ".pdf":
        return pdf_blocks(data)
    if extension in {".docx", ".xlsx", ".pptx"}:
        return office_blocks(extension, data)
    if extension in {".html", ".htm"}:
        return html_blocks(data)
    if extension in {".md", ".markdown", ".txt"}:
        return [Block(decode(data))]
    raise ValueError("支持 PDF、DOCX、XLSX、PPTX、HTML、Markdown 和 TXT。")


def clean(blocks, *, min_chars=1):
    for block in blocks:
        text = unicodedata.normalize("NFC", block.text).replace("\r\n", "\n")
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f\u200b\ufeff]", "", text)
        text = re.sub(r"(?m)^[ \t]+$|[ \t]+$", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if sum(ch.isalnum() for ch in text) >= min_chars:
            yield Block(text, block.location, block.kind, block.level)


def windows(text, size, overlap):
    for start in range(0, len(text), size - overlap):
        if start and start + overlap >= len(text):
            break
        yield text[start:start + size]


def split_blocks(blocks, strategy, size, overlap, embed):
    headings = {}
    for block in blocks:
        if strategy == "fixed":
            yield from (Block(text, block.location, block.kind) for text in windows(block.text, size, overlap))
            continue
        paragraphs = re.split(r"\n\s*\n|(?=^#{1,6}\s)", block.text, flags=re.M)
        for paragraph in filter(str.strip, paragraphs):
            title = re.match(r"^(#{1,6})[ \t]+([^\n]+)", paragraph) if block.kind == "text" else None
            level = block.level or (len(title[1]) if title else 0)
            prefix = ""
            if strategy == "hierarchical":
                if level:
                    headings = {depth: heading for depth, heading in headings.items() if depth < level}
                breadcrumb = " > ".join(headings.values())
                if breadcrumb:
                    prefix = breadcrumb[:size // 3] + "\n"
                if level:
                    headings[level] = title[2].strip() if title else paragraph.splitlines()[0]
            if strategy == "semantic" and block.kind == "text":
                sentences = [s for s in re.split(r"(?<=[。！？!?])|(?<=\.)\s+|\n+", paragraph) if s.strip()]
                if len(sentences) > 1:
                    vectors = embed([sentence[:size] for sentence in sentences])
                    groups = [sentences[0]]
                    for i, sentence in enumerate(sentences[1:], 1):
                        left, right = vectors[i - 1], vectors[i]
                        norm = hypot(*left) * hypot(*right)
                        similar = sum(a * b for a, b in zip(left, right)) / norm if norm else 0
                        if similar < 0.65 or len(groups[-1]) + len(sentence) + 1 > size:
                            groups.append(sentence)
                        else:
                            groups[-1] += " " + sentence
                    for group in groups:
                        yield from (Block(text, block.location, block.kind) for text in windows(group, size, overlap))
                    continue
            yield from (Block(prefix + text, block.location, block.kind)
                        for text in windows(paragraph, size - len(prefix), min(overlap, size - len(prefix) - 1)))


def parent_child_chunks(blocks, embed, parent_size=900, child_size=250, overlap=50):
    parents = []
    for block in split_blocks(blocks, "hierarchical", parent_size, overlap, embed):
        previous = parents[-1] if parents else None
        if (previous and previous.location == block.location and previous.kind == block.kind == "text"
                and len(previous.text) + len(block.text) + 2 <= parent_size):
            parents[-1] = Block(previous.text + "\n\n" + block.text, block.location, block.kind)
        else:
            parents.append(block)
    position = 0
    for parent in parents:
        parent_position = position
        yield parent, "parent", None
        position += 1
        for text in windows(parent.text, child_size, overlap):
            yield Block(text, parent.location, parent.kind), "child", parent_position
            position += 1
