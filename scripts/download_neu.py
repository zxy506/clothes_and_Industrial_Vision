"""Download NEU-CLS through the link published on its official university page."""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import requests


FILE_ID = "1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc"
OFFICIAL_PAGE = "https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
DEST = Path("data/raw/neu_cls/NEU-CLS.rar")


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        if tag != "input":
            return
        attrs = dict(attrs)
        if attrs.get("type") == "hidden" and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")


def main():
    if DEST.exists():
        raise FileExistsError(f"archive already exists; will not overwrite: {DEST}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temp = DEST.with_suffix(".rar.part")
    if temp.exists():
        raise FileExistsError(f"partial download exists; inspect before retry: {temp}")
    with requests.Session() as session:
        warning = session.get(URL, timeout=30)
        warning.raise_for_status()
        form = DownloadForm()
        form.feed(warning.text)
        if form.fields.get("id") != FILE_ID or form.fields.get("confirm") != "t":
            raise RuntimeError("official Google Drive confirmation form not found")
        with session.get("https://drive.usercontent.google.com/download",
                         params=form.fields, stream=True, timeout=60) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", ""):
                raise RuntimeError("download returned HTML instead of the archive")
            digest = hashlib.sha256()
            size = 0
            try:
                with temp.open("xb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            out.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                expected = response.headers.get("content-length")
                if expected and size != int(expected):
                    raise RuntimeError(f"truncated archive: {size} != {expected}")
                temp.replace(DEST)
            except Exception:
                temp.unlink(missing_ok=True)
                raise
    provenance = {
        "official_page": OFFICIAL_PAGE, "file_id": FILE_ID,
        "archive": str(DEST), "bytes": size, "sha256": digest.hexdigest(),
        "license_note": "Official page does not state a complete redistribution license."
    }
    Path("data/raw/neu_cls/source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
