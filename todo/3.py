import subprocess
from pathlib import Path
import sys

qpdf_out = base_dir / "tmp_qpdf.pdf"
output_pdf = base_dir / "output.pdf"

# 当前目录
base_dir = Path(__file__).parent.resolve()

# 输入/输出 PDF
input_pdf = (
    base_dir
    / "LiuJiyang_Sun Yat-sen University_Bachelor’s Degree Certificate_2025-06-23-TRANSLATED.pdf"
)

# 路径
qpdf_path = r"C:\2SilverHand\qpdf\bin\qpdf.exe"
gs_path = r"C:\2SilverHand\ghostpdl\gs10.05.1\bin\gswin64c.exe"


def run(cmd, title):
    print(f"\n=== {title} ===")
    print(" ".join(str(x) for x in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", res.stdout.strip())
    print("STDERR:\n", res.stderr.strip())
    if res.returncode != 0:
        print(f"{title} failed with code {res.returncode}")
        sys.exit(res.returncode)


# Step 1: qpdf 压缩结构
qpdf_cmd = [
    qpdf_path,
    "--warning-exit-0",
    "--stream-data=compress",
    "--object-streams=generate",
    "--optimize-images",
    "--jpeg-quality=5",  # JPEG 重压缩，但不降采样
    str(input_pdf),
    str(qpdf_out),
]
run(qpdf_cmd, "QPDF pass")

# Step 2: Ghostscript 降采样 + 压缩
# 预设档位: /screen=72dpi (最小), /ebook=150dpi (推荐), /printer=300dpi
# preset = "/ebook"
preset = "/prepress"

gs_cmd = [
    gs_path,
    "-sDEVICE=pdfwrite",
    "-dCompatibilityLevel=1.4",
    f"-dPDFSETTINGS={preset}",
    "-dNOPAUSE",
    "-dQUIET",
    "-dBATCH",
    "-sOutputFile=" + str(output_pdf),
    str(qpdf_out),
]
run(gs_cmd, f"Ghostscript downsample {preset}")

print("\nDone! Compressed file:", output_pdf.resolve())
