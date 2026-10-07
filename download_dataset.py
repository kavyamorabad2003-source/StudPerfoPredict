"""Downloads the real UCI Student Performance dataset into ./data/student-mat.csv"""
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DATA = Path(__file__).parent / "data"


def extract(zf: zipfile.ZipFile):
    for name in zf.namelist():
        if name.lower().endswith(".zip"):
            extract(zipfile.ZipFile(io.BytesIO(zf.read(name))))
        elif name.lower().endswith((".csv", ".txt")):
            target = DATA / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print("extracted", target.name)


def main():
    DATA.mkdir(exist_ok=True)
    print("Downloading", URL)
    raw = urllib.request.urlopen(URL, timeout=60).read()
    extract(zipfile.ZipFile(io.BytesIO(raw)))
    if (DATA / "student-mat.csv").exists():
        print("Done. Restart the app to use the real dataset.")
    else:
        print("student-mat.csv not found. Download it manually from "
              "https://archive.ics.uci.edu/dataset/320/student+performance")


if __name__ == "__main__":
    main()
