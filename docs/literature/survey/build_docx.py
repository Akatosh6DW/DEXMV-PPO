"""从 main.tex 生成 Word 版综述（公式为 Word 原生公式，参考文献与 PDF 逐条一致）。

前置：先用 XeLaTeX/tectonic 编译 main.tex，得到 main.aux 与 main.bbl；需要 pandoc。
用法：python build_docx.py --pandoc PANDOC --fig fig1.png [--out 综述.docx]
"""
import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_labels(aux):
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux):
        labels[m.group(1)] = m.group(2)
    return labels


def read_bbl(bbl):
    """返回 [(key, 纯文本条目)]，顺序即编号顺序。"""
    entries = []
    for m in re.finditer(r"\\bibitem(?:\[.*?\])?\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})", bbl, re.S):
        key, body = m.group(1), m.group(2)
        body = re.sub(r"\\allowbreak\s*", "", body.replace("\\newblock", " "))
        body = re.sub(r"\\doi\{([^}]*)\}", r"\1", body)
        body = re.sub(r"\\url\{([^}]*)\}", r"\1", body)
        body = body.replace("~", " ").replace("\\&", "&")
        body = body.replace("{", "").replace("}", "")
        body = re.sub(r"\s+", " ", body).strip()
        entries.append((key, body))
    return entries


def fmt_nums(nums):
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}-{nums[j]}" if j - i >= 2 else ",".join(map(str, nums[i:j + 1])))
        i = j + 1
    return "[" + ",".join(parts) + "]"


def tex_escape(s):
    return s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")


# 按列数区分三张表的列宽比例（表 1 五列、表 2 四列、表 3 六列）
COL_WIDTHS = {5: [0.10, 0.18, 0.28, 0.34, 0.10], 4: [0.17, 0.25, 0.25, 0.33],
              6: [0.12, 0.15, 0.17, 0.17, 0.20, 0.19]}


def preprocess(tex, labels, refs):
    num = {k: i + 1 for i, (k, _) in enumerate(refs)}
    body = tex[tex.index("\\begin{document}") + len("\\begin{document}"):tex.index("\\end{document}")]

    # 图 1：TikZ 换为图片
    body = re.sub(r"\\begin\{figure\}.*?\\caption\{([^}]*)\}\\label\{([^}]*)\}\s*\\end\{figure\}",
                  lambda m: "\\begin{figure}\n\\includegraphics{fig1.png}\n\\caption{图 %s　%s}\n\\end{figure}"
                  % (labels[m.group(2)], m.group(1)), body, flags=re.S)
    # 表：编号写进标题；tabularx → tabular
    body = re.sub(r"\\caption\{([^}]*)\}\\label\{(tab:[^}]*)\}",
                  lambda m: "\\caption{表 %s　%s}" % (labels[m.group(2)], m.group(1)), body)

    def to_tabular(m):
        rows = m.group(1)
        ncol = rows.split("\\\\")[0].count("&") + 1
        widths = COL_WIDTHS.get(ncol, [1 / ncol] * ncol)
        spec = "".join("p{%.2f\\linewidth}" % w for w in widths)
        return "\\begin{tabular}{%s}%s\\end{tabular}" % (spec, rows)
    body = re.sub(r"\\begin\{tabularx\}\{\\textwidth\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabularx\}",
                  to_tabular, body, flags=re.S)
    body = body.replace("\\-", "")

    # 公式：去掉 label，把编号写进公式右侧
    def eqn(m):
        content, label = m.group(2), m.group(1)
        return "\\begin{equation}%s\\qquad(%s)\n\\end{equation}" % (content.rstrip(), labels[label])
    body = re.sub(r"\\begin\{equation\}\\label\{([^}]*)\}(.*?)\\end\{equation\}", eqn, body, flags=re.S)

    body = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(%s)" % labels[m.group(1)], body)
    body = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], body)
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = re.sub(r"\\cite\{([^}]*)\}",
                  lambda m: "\\textsuperscript{%s}" % fmt_nums([num[k.strip()] for k in m.group(1).split(",")]), body)

    # 摘要区的字体命令
    body = re.sub(r"\{\\heiti ([^}]*)\}", r"\\textbf{\1}", body)
    body = re.sub(r"\{\\large\\bfseries ([^}]*)\}", r"\\textbf{\1}", body)
    for cmd in ("\\begingroup\\small", "\\endgroup", "\\smallskip", "\\bigskip", "\\noindent"):
        body = body.replace(cmd, "")

    ref_tex = "\\section*{参考文献}\n\n" + "\n\n".join("[%d] %s" % (i + 1, tex_escape(t)) for i, (_, t) in enumerate(refs))
    body = body.replace("\\bibliography{refs}", ref_tex)

    title = re.search(r"\\title\{\\heiti (.*?)\}\n", tex).group(1)
    head = ("\\documentclass{article}\n\\title{%s}\n\\author{（作者姓名）　（作者单位）}\n\\date{2026 年 10 月}\n"
            "\\begin{document}\n" % title)
    return head + body + "\n\\end{document}\n"


# ---------- Word 样式（中文排版习惯） ----------
FONT_SONG = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" w:cs="Times New Roman"/>'
FONT_HEI = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体" w:cs="Times New Roman"/>'


def pstyle(sid, name, based, ppr, rpr, extra=""):
    return (f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{sid}"><w:name w:val="{name}"/>'
            f'<w:basedOn w:val="{based}"/>{extra}<w:qFormat/><w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>')


STYLES = {
    "Normal": ('<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/>'
               '<w:pPr><w:spacing w:before="0" w:after="0" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
               f'<w:rPr>{FONT_SONG}<w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>'),
    "BodyText": pstyle("BodyText", "Body Text", "Normal", '<w:ind w:firstLineChars="200" w:firstLine="480"/>', ""),
    "FirstParagraph": pstyle("FirstParagraph", "First Paragraph", "BodyText", "", "", '<w:next w:val="BodyText"/>'),
    "Compact": pstyle("Compact", "Compact", "Normal", '<w:ind w:firstLine="0"/>', ""),
    "Title": pstyle("Title", "Title", "Normal", '<w:spacing w:before="240" w:after="240"/><w:jc w:val="center"/>',
                    f'{FONT_HEI}<w:b/><w:sz w:val="36"/><w:szCs w:val="36"/>'),
    "Author": pstyle("Author", "Author", "Normal", '<w:jc w:val="center"/>', '<w:sz w:val="24"/>'),
    "Date": pstyle("Date", "Date", "Normal", '<w:spacing w:after="240"/><w:jc w:val="center"/>', '<w:sz w:val="24"/>'),
    "Abstract": pstyle("Abstract", "Abstract", "Normal", '<w:ind w:firstLine="0"/>', '<w:sz w:val="21"/><w:szCs w:val="21"/>'),
    "Heading1": pstyle("Heading1", "heading 1", "Normal",
                       '<w:keepNext/><w:spacing w:before="360" w:after="120"/><w:jc w:val="left"/><w:outlineLvl w:val="0"/>',
                       f'{FONT_HEI}<w:b/><w:color w:val="000000"/><w:sz w:val="28"/><w:szCs w:val="28"/>'),
    "Heading2": pstyle("Heading2", "heading 2", "Normal",
                       '<w:keepNext/><w:spacing w:before="240" w:after="60"/><w:jc w:val="left"/><w:outlineLvl w:val="1"/>',
                       f'{FONT_HEI}<w:b/><w:color w:val="000000"/><w:sz w:val="24"/><w:szCs w:val="24"/>'),
    "Caption": pstyle("Caption", "caption", "Normal", '<w:spacing w:before="120" w:after="120"/><w:jc w:val="center"/>',
                      f'{FONT_HEI}<w:sz w:val="21"/><w:szCs w:val="21"/>'),
    "TableCaption": pstyle("TableCaption", "Table Caption", "Caption", '<w:keepNext/>', ""),
    "ImageCaption": pstyle("ImageCaption", "Image Caption", "Caption", "", ""),
    "Figure": pstyle("Figure", "Figure", "Normal", '<w:jc w:val="center"/>', ""),
    "CaptionedFigure": pstyle("CaptionedFigure", "Captioned Figure", "Figure", '<w:keepNext/><w:jc w:val="center"/>', ""),
    "Bibliography": pstyle("Bibliography", "Bibliography", "Normal",
                           '<w:ind w:left="480" w:hanging="480"/><w:jc w:val="left"/>', '<w:sz w:val="21"/><w:szCs w:val="21"/>'),
    "TableText": pstyle("TableText", "Table Text", "Normal",
                        '<w:spacing w:line="276" w:lineRule="auto"/><w:jc w:val="left"/><w:ind w:firstLine="0"/>',
                        '<w:sz w:val="21"/><w:szCs w:val="21"/>'),
    "Table": ('<w:style w:type="table" w:default="1" w:styleId="Table"><w:name w:val="Table"/><w:basedOn w:val="TableNormal"/><w:qFormat/>'
              '<w:pPr><w:spacing w:line="300" w:lineRule="auto"/><w:jc w:val="left"/></w:pPr><w:rPr><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
              '<w:tblPr><w:jc w:val="center"/><w:tblBorders><w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
              '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/></w:tblBorders>'
              '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/>'
              '<w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr>'
              '<w:tblStylePr w:type="firstRow"><w:rPr><w:b/></w:rPr><w:tcPr><w:tcBorders>'
              '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders></w:tcPr></w:tblStylePr></w:style>'),
}


def make_reference_docx(pandoc, dst):
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "ref.docx"
    with open(raw, "wb") as f:
        subprocess.run([pandoc, "--print-default-data-file", "reference.docx"], stdout=f, check=True)
    zin = zipfile.ZipFile(raw)
    styles = zin.read("word/styles.xml").decode("utf-8")
    styles = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>",
                    f"<w:rPrDefault><w:rPr>{FONT_SONG}<w:sz w:val=\"24\"/><w:szCs w:val=\"24\"/>"
                    "<w:lang w:val=\"en-US\" w:eastAsia=\"zh-CN\" w:bidi=\"ar-SA\"/></w:rPr></w:rPrDefault>", styles, flags=re.S)
    styles = re.sub(r"<w:pPrDefault>.*?</w:pPrDefault>", "<w:pPrDefault><w:pPr/></w:pPrDefault>", styles, flags=re.S)
    for sid, xml in STYLES.items():
        pat = re.compile(r'<w:style [^>]*w:styleId="%s"[^>]*>.*?</w:style>' % sid, re.S)
        styles, n = pat.subn(lambda _: xml, styles)
        if n == 0:
            styles = styles.replace("</w:styles>", xml + "</w:styles>")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = styles.encode("utf-8") if item.filename == "word/styles.xml" else zin.read(item.filename)
            zout.writestr(item, data)
    shutil.rmtree(tmp)


def postprocess(docx):
    """摘要区用 Abstract 样式；参考文献用悬挂缩进；页面 A4、2.5 cm 边距。"""
    tmp = Path(str(docx) + ".tmp")
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                paras = re.split(r"(?=<w:p>|<w:p )", xml)
                out, zone = [], "front"
                for p in paras:
                    if 'w:val="Heading1"' in p:
                        zone = "refs" if "参考文献" in re.sub(r"<[^>]+>", "", p) else "body"
                    elif zone in ("front", "refs") and p.startswith("<w:p") and 'w:val="Title"' not in p \
                            and 'w:val="Author"' not in p and 'w:val="Date"' not in p:
                        style = "Abstract" if zone == "front" else "Bibliography"
                        if "<w:pStyle " in p:
                            p = re.sub(r'<w:pStyle w:val="[^"]*"\s*/>', f'<w:pStyle w:val="{style}"/>', p, count=1)
                        elif "<w:pPr>" in p:
                            p = p.replace("<w:pPr>", f'<w:pPr><w:pStyle w:val="{style}"/>', 1)
                        else:
                            p = re.sub(r"^(<w:p[^>]*>)", r'\1<w:pPr><w:pStyle w:val="%s"/></w:pPr>' % style, p, count=1)
                    out.append(p)
                xml = "".join(out)
                # 表格单元格用 TableText；标题编号后用空格代替制表符；英文题目居中
                xml = re.sub(r"<w:tbl>.*?</w:tbl>",
                             lambda m: m.group(0).replace('w:val="Compact"', 'w:val="TableText"'), xml, flags=re.S)
                xml = re.sub(r'(<w:pStyle w:val="Heading[12]" />.*?)<w:r><w:tab /></w:r>',
                             r'\1<w:r><w:t xml:space="preserve"> </w:t></w:r>', xml)
                i = xml.find("A Survey on")
                j = xml.rfind('<w:pStyle w:val="Abstract"/>', 0, i)
                xml = xml[:j] + '<w:pStyle w:val="Abstract"/><w:jc w:val="center"/>' + xml[j + len('<w:pStyle w:val="Abstract"/>'):]
                sect = ('<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1417" w:right="1417" w:bottom="1417" '
                        'w:left="1417" w:header="851" w:footer="992" w:gutter="0"/>')
                xml = re.sub(r"<w:pgSz[^>]*/>", "", xml)
                xml = re.sub(r"<w:pgMar[^>]*/>", "", xml)
                xml = re.sub(r"(<w:sectPr[^>]*>)", r"\1" + sect, xml, count=1)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(docx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pandoc", default="pandoc")
    ap.add_argument("--fig", default=str(HERE / "fig1.png"))
    ap.add_argument("--out", default=str(HERE / "基于人类视频先验的灵巧手强化学习研究综述.docx"))
    a = ap.parse_args()

    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    labels = read_labels((HERE / "main.aux").read_text(encoding="utf-8"))
    refs = read_bbl((HERE / "main.bbl").read_text(encoding="utf-8"))
    work = Path(tempfile.mkdtemp())
    (work / "src.tex").write_text(preprocess(tex, labels, refs), encoding="utf-8")
    shutil.copy(a.fig, work / "fig1.png")
    make_reference_docx(a.pandoc, work / "reference.docx")
    subprocess.run([a.pandoc, "src.tex", "-f", "latex", "-t", "docx", "--number-sections",
                    "--reference-doc=reference.docx", "-o", "out.docx"], cwd=work, check=True)
    postprocess(work / "out.docx")
    shutil.copy(work / "out.docx", a.out)
    shutil.rmtree(work)
    print("written", a.out, "refs", len(refs))


if __name__ == "__main__":
    main()
