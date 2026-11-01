from smartsearch.extractors import extract_text


def test_docx(tmp_path):
    import docx

    p = tmp_path / "a.docx"
    d = docx.Document()
    d.add_paragraph("Quarterly sales review")
    d.save(p)
    assert "Quarterly sales review" in extract_text(p)


def test_xlsx(tmp_path):
    from openpyxl import Workbook

    p = tmp_path / "a.xlsx"
    wb = Workbook()
    wb.active.append(["region", "revenue"])
    wb.active.append(["EMEA", 42])
    wb.save(p)
    text = extract_text(p)
    assert "EMEA" in text and "42" in text


def test_html_strips_scripts(tmp_path):
    p = tmp_path / "a.html"
    p.write_text("<html><script>var x=1</script><body><p>Hello page</p></body></html>")
    text = extract_text(p)
    assert "Hello page" in text and "var x" not in text


def test_binary_file_returns_empty(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"\x00\x01\x02binary")
    assert extract_text(p) == ""
