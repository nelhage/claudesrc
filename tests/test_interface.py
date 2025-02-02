from scrubs import prompts
from scrubs.interface import format_text


def test_format_text():
    assert format_text("assistant", "some text") == "some text"

    contents = "".join([f"{i:03d}\n" for i in range(10)])

    rendered = prompts.file_contents("a/file.txt", contents)

    formatted = format_text("user", rendered)

    assert formatted.count("\n") < 5
    assert "10 lines" in formatted
