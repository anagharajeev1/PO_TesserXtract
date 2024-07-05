"""
This Flask application leverages OCR and regular expressions to seamlessly extract valuable information from image files. Key technical features include:

- OCR Engine Integration: Employs Tesseract, a robust open-source OCR engine, to accurately extract text from uploaded image files.
- Text Extraction and Data Framing: Transforms extracted text into structured JSON format, making it readily consumable for further analysis or integration with other systems.
- Precise Field Extraction: Utilizes meticulously crafted regular expressions to pinpoint and isolate specific fields of interest, streamlining data retrieval.
- Multiprocessing Capabilities: Enhances efficiency by supporting concurrent processing of multiple image uploads, enabling users to process larger batches of files effectively.

Additional Technical Considerations:

- Python Foundation: Built upon the versatile Python programming language, known for its extensive libraries and readability.
- Flask Framework: Utilizes the lightweight and flexible Flask web framework for streamlined web application development.
- Error Handling: Incorporates robust error handling mechanisms to gracefully manage potential file upload issues, OCR processing errors, or invalid field extraction scenarios.
- Asynchronous Processing: Explores integration with asynchronous task queues for further performance optimization, especially when handling large-scale image uploads or computationally intensive OCR tasks.

Author:
    Girwar Singh Bhati (girwarsinghbhati1@gmail.com)
    
Example Usage:
- Upload a file to the Flask application.
- The file is processed using OCR and regular expressions to extract field values.
- The extracted field values are returned as JSON.

Example input:
- File: invoice.png

Example output:
{
  "fields": {
    "invoice.png": {
      "GSTIN": "ABCD1234",
      "Invoice Number": "INV-001",
      "Invoice Date": "01 Jan 2022",
      ...
    }
  }
}

Inputs:
- File(s) to be uploaded and processed by the Flask application.

Flow:
1. The Flask application is created and configured.
2. The `/upload` route is defined to handle file uploads.
3. The uploaded files are checked for validity (image file and size limit).
4. The files are saved to the `UPLOAD_FOLDER` directory with unique filenames.
5. The files are processed using multiprocessing, where each file is passed to the `process_file` function.
6. The `process_file` function performs OCR on the image file and extracts field values using regular expressions.
7. The extracted field values are returned along with the filename.
8. The extracted field values and filenames are stored in a dictionary.
9. The dictionary is returned as JSON response.

Outputs:
- JSON response containing the extracted field values and filenames.
"""
import os
import re
import cv2
import numpy as np
import multiprocessing as mp
import uuid
import pytesseract
from flask import Flask, render_template, request, jsonify

# create a Flask app instance
app = Flask(__name__)

# configure the app
app.config['UPLOAD_FOLDER'] = 'input'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
app.config['MAX_FILES'] = 10  # maximum number of files to process at once


# Define regex patterns for extraction
# patterns = {
#     "PR Number": r"PR Number\s*(\d+)",
#     "Technical Specifications": r"Technical Specifications\s*:\s*([A-Za-z0-9\s:]+)",
#     "Vendor Name": r"Vendor Name\s*([A-Za-z]+)",
#     "Vendor Payment Terms": r"Vendor Payment Terms\s*([\d%a-zA-Z\s,]+)",
#     "Vendor Delivery Terms": r"Vendor Delivery Terms\s*([\w\s\-:,]+)",
#     "Cost Estimate": r"Cost Estimate\s*(\d+)",
#     "Budget Allocated": r"Budget Allocated\s*(\d+)"
# }
# Define regex patterns for extraction
patterns = {
    "PR Number": r"PR Number\s*(\d+)",
    "Technical Specifications": r"Technical Specifications\s*([%A-Za-z\s,]+)",
    "Vendor Name": r"Vendor Name\s*([A-Za-z]+)",
    "Vendor Payment Terms": r"Vendor Payment Terms\s*([%A-Za-z\s,]+)",
    "Vendor Delivery Terms": r"Vendor Delivery Terms\s*([%A-Za-z\s,\d]+)",
    "Cost Estimate": r"Cost Estimate\s*(\d+)",
    "Budget Allocated": r"Budget Allocated\s*(\d+)"
}

def is_image_file(filename):
    """Check if a file is an image file based on its extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg', 'gif','pdf']

def generate_unique_filename(filename):
    """Generate a unique filename for an uploaded file."""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    new_filename = f"{str(uuid.uuid4())}.{ext}"
    return new_filename

def clean_text(text):
    """Clean and format extracted text."""
    return re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace and line breaks

def process_file(input_image_path):
    """Perform OCR on an image file and extract field values using regular expressions."""
    image = cv2.imread(input_image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Perform OCR on the processed image
    invoice_text = pytesseract.image_to_string(thresh)
    
    # Initialize empty dictionary to store field values
    fields = {}

    # Iterate through each pattern and extract its corresponding value
    for key, pattern in patterns.items():
        match = re.search(pattern, invoice_text, re.IGNORECASE)  # Ignore case for flexibility
        if match:
            fields[key] = clean_text(match.group(1))

    # Return the extracted fields and the file name
    if not fields:
        return os.path.basename(input_image_path), 'Not an Invoice Image'
    else:
        return os.path.basename(input_image_path), fields

@app.route('/')
def index():
    """Render the index.html template."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file uploads and process them using multiprocessing."""
    # Check if any files were submitted
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files selected for upload'})

    # Get the list of files submitted
    files = request.files.getlist('files[]')

    # Check if the number of files is within the allowed limit
    if len(files) > app.config['MAX_FILES']:
        return jsonify({'error': f"Maximum {app.config['MAX_FILES']} files are allowed at a time"})

    # Check if each file is an image file and within the size limit
    for file in files:
        if not is_image_file(file.filename):
            return jsonify({'error': f"{file.filename} is not an image file"})

        if file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': f"{file.filename} is too large (max size 5MB)"})

    # Create the input directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Save the files to the input directory and create a list of their filenames
    filenames = []
    for file in files:
        filename = generate_unique_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filenames.append(filename)

    # Process the files using multiprocessing
    pool = mp.Pool(processes=len(filenames))
    results = [pool.apply_async(process_file, args=(os.path.join(app.config['UPLOAD_FOLDER'], filename),)) for filename in filenames]
    fields_list = [result.get() for result in results]
    pool.close()
    pool.join()

    # Create a dictionary of extracted fields and their associated file names
    response = {}
    for i in range(len(filenames)):
        response[filenames[i]] = fields_list[i][1]

    # Return the extracted fields and file names as JSON
    return jsonify({'fields': response})

if __name__ == '__main__':
    app.run(debug=True)
