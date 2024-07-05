import os
import re
import cv2
import multiprocessing as mp
import uuid
import pytesseract
from flask import Flask, render_template, request, jsonify
from views.regex_pattern import patterns


# create a Flask app instance
app = Flask(__name__)

# configure the app
app.config['UPLOAD_FOLDER'] = 'input'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
app.config['MAX_FILES'] = 10  # maximum number of files to process at once


# -----------------------------------------------------------------------------------------------------
def is_image_file(filename):
    """Check if a file is an image file based on its extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg', 'gif']


# -----------------------------------------------------------------------------------------------------
def generate_unique_filename(filename):
    """Generate a unique filename for an uploaded file."""
    # get the file extension
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    # generate a random UUID and use it as the new filename
    new_filename = f"{str(uuid.uuid4())}.{ext}"
    return new_filename


# -----------------------------------------------------------------------------------------------------
def process_file(input_image_path):
    """Perform OCR on an image file and extract field values using regular expressions."""
    image = cv2.imread(input_image_path)
    invoice_text = pytesseract.image_to_string(image)

    # initialize empty dictionary to store field values
    fields = {}

    # iterate through each pattern and extract its corresponding value
    for key, pattern in patterns.items():
        match = re.search(pattern, invoice_text)
        if match:
            fields[key] = match.group(1).strip()
    print(fields)

    # return the extracted fields and the file name

    if not fields:
        return os.path.basename(input_image_path), 'Not an Invoice Image'
    else:
        # return the extracted fields and the file name
        return os.path.basename(input_image_path), fields



# -----------------------------------------------------------------------------------------------------
@app.route('/')
def index():
    """Render the index.html template."""
    return render_template('index.html')



# -----------------------------------------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    """Handle file uploads and process them using multiprocessing."""
    # check if any files were submitted
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files selected for upload'})

    # get the list of files submitted
    files = request.files.getlist('files[]')

    # check if the number of files is within the allowed limit
    if len(files) > app.config['MAX_FILES']:
        return jsonify({'error': f"Maximum {app.config['MAX_FILES']} files are allowed at a time"})

    # check if each file is an image file and within the size limit
    for file in files:
        if not is_image_file(file.filename):
            return jsonify({'error': f"{file.filename} is not an image file"})

        if file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': f"{file.filename} is too large (max size 5MB)"})

    # create the input directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # save the files to the input directory and create a list of their filenames
    filenames = []
    for file in files:
        filename = generate_unique_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filenames.append(filename)


    # process the files using multiprocessing
    pool = mp.Pool(processes=len(filenames))
    results = [pool.apply_async(process_file, args=(os.path.join(app.config['UPLOAD_FOLDER'], filename),)) for filename in filenames]
    fields_list = [result.get() for result in results]
    pool.close()
    pool.join()

    # create a dictionary of extracted fields and their associated file names
    response = {}
    for i in range(len(filenames)):
        response[filenames[i]] = fields_list[i][1]

    # return the extracted fields and file names as JSON
    return jsonify({'fields': response})

# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)

