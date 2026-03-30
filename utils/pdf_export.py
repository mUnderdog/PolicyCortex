from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
import tempfile
import os


def generate_pdf(title: str, content: str) -> str:
    """
    Generate a PDF file from a title and multi-line content string.
    Returns the path to the temporary PDF file.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.close()  # Close so ReportLab can write to it on Windows

    c = canvas.Canvas(temp_file.name, pagesize=letter)
    width, height = letter

    margin = 50
    max_width = width - 2 * margin  # usable text width
    y = height - margin

    # --- Title ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 30

    # Divider line
    c.setLineWidth(0.5)
    c.line(margin, y, width - margin, y)
    y -= 20

    # --- Body content (with word-wrap) ---
    c.setFont("Helvetica", 11)
    line_height = 16

    for paragraph in content.split("\n"):
        # Wrap long lines to fit within the page width
        wrapped_lines = simpleSplit(paragraph, "Helvetica", 11, max_width)

        # Treat blank paragraphs as a blank line
        if not wrapped_lines:
            wrapped_lines = [""]

        for line in wrapped_lines:
            if y < margin + line_height:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - margin

            c.drawString(margin, y, line)
            y -= line_height

    c.save()
    return temp_file.name