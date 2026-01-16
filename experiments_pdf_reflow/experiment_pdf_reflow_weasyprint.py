from weasyprint import HTML  
from spire.pdf.common import *
from spire.pdf import *

def convert_pdf_to_html(input_pdf_path, output_html_path):
    """
    Converts an entire PDF document to an HTML file.

    Args:
        input_pdf_path (str): The path to the input PDF file.
        output_html_path (str): The path where the output HTML file will be saved.
    """
    print(f"Converting '{input_pdf_path}' to '{output_html_path}'...")

    # Create a PDF document object
    doc = PdfDocument()

    try:
        # Load the PDF file from the specified path
        doc.LoadFromFile(input_pdf_path)

        # Convert the loaded PDF document to HTML format and save it
        # FileFormat.HTML is the enumeration specifying the output format
        doc.SaveToFile(output_html_path, FileFormat.HTML)

        print("Conversion successful!")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
    finally:
        # Always close the document to release resources
        doc.Close()



convert_pdf_to_html('FakeStackedEmailThread.pdf', 'test_html.html')

html_content = HTML('test_html.html')
html_content.write_pdf('pdf_to_html_to_pdf_demo.pdf')