import html


def file_contents(path: str, contents: str) -> str:
    bits = []
    bits.append(f"<file path='{html.escape(path)}'>\n")
    bits.append(contents)
    if not contents.endswith("\n"):
        bits.append("\n")
    bits.append("</file>\n")
    return "".join(bits)
