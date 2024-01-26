import glob
import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from PyPDF2 import PdfReader
from pdf2image import convert_from_path


def crop_page_to_columns(pdf_path, output_folder, header_height, footer_height, column_offset, start_page=0, end_page=None):
    """
    This function crops each page of a PDF file into two columns and saves them as separate images.
    It first clears the output folder, then determines the total number of pages in the PDF.
    If the end_page parameter is not specified or exceeds the number of pages in the PDF, it is adjusted to the total number of pages.
    If the output folder does not exist, it is created.
    The PDF is then converted to images using a DPI of 300, and 10 pages are processed at a time.
    For each page, the dimensions for cropping are calculated based on the specified header and footer heights, and the column offset.
    The left and right columns are cropped and saved as separate images in the output folder.
    The function prints a message indicating the page number that has been processed and cropped into two columns.
    The start_page parameter is incremented to process the next set of pages, and the loop continues until there
        are no more pages left to process or the end_page is reached.

    :param: pdf_path (str): Path to the PDF file to be processed.
    :param: output_folder (str): Path to the folder where the cropped images will be saved.
    :param: header_height (int): Height of the header to be excluded from cropping.
    :param: footer_height (int): Height of the footer to be excluded from cropping.
    :param: column_offset (int): Offset to adjust the position of the middle column.
    :param: start_page (int, optional): The first page number to start cropping from. Defaults to 0.
    :param: end_page (int, optional): The last page number to stop cropping at. If None, it will crop until the last page of the PDF. Defaults to None.

    :return: None
    """
    # Clear the output folder before processing new files
    clear_output_folder(output_folder)

    # Determine the total number of pages in the PDF
    with open(pdf_path, 'rb') as f:
        pdf = PdfReader(f)
        total_pages = len(pdf.pages)

    # Adjust the end_page if it is beyond the actual number of pages
    if end_page is None or end_page > total_pages:
        end_page = total_pages
        print(f"Adjusted the last page to the total number of pages in the PDF: {total_pages}")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert the PDF to images (10 pages at a time)
    while True:
        pages = convert_from_path(pdf_path, dpi=300, first_page=start_page + 1, last_page=min(end_page, start_page + 10) if end_page else start_page + 10)
        if not pages:
            break  # Break the loop if there are no pages left to process

        for i, page_image in enumerate(pages, start=start_page):
            # Calculate the dimensions for cropping
            width, height = page_image.size
            top = header_height
            bottom = height - footer_height
            middle = (width // 2) + column_offset

            # Crop out the left column
            left_column = page_image.crop((0, top, middle, bottom))
            left_filename = f"page_{i + 1:04d}_left.png"
            left_column.save(os.path.join(output_folder, left_filename))

            # Crop out the right column
            right_column = page_image.crop((middle, top, width, bottom))
            right_filename = f"page_{i + 1:04d}_right.png"
            right_column.save(os.path.join(output_folder, right_filename))

            print(f"Processed page {i + 1}, cropped into two columns")

        start_page += 10  # Increment the start_page to process the next set of pages

        if end_page and start_page >= end_page:
            break  # Break the loop if the end_page is reached


def extract_text_and_append_to_file(images_folder, output_text_file):
    """
    Extract text from each image in the specified folder using Tesseract OCR and append it to the specified text file.
    Processes each image file in the folder, sorted by numeric order. If text is successfully extracted, it's appended to the output file.
    Each image is tried twice for text extraction before moving on to the next image.

    :param: images_folder: Folder containing the images to be processed.
    :param: output_text_file: File where the extracted text will be appended.

    :return: None
    """
    # List all image files in the specified folder, sorted by numeric order
    image_files = sorted(
        [f for f in os.listdir(images_folder) if f.endswith('.png')],
        key=lambda x: (int(x.split('_')[1]), x.split('_')[2])
    )

    # Open the output file in append mode
    with open(output_text_file, 'a') as file:
        # Process each image file
        for image_file in image_files:
            # Full path to the image file
            image_path = os.path.join(images_folder, image_file)

            for attempt in range(2):  # Allow for one retry
                image = Image.open(image_path)
                # Preprocess the image for OCR
                processed_image = preprocess_image_for_ocr(image)
                # Extract text from the image using Tesseract
                extracted_text = pytesseract.image_to_string(processed_image, config='--psm 6')

                # If text was extracted successfully, break out of the retry loop
                if extracted_text.strip():  # Checks if the text is not just whitespace
                    # Append the extracted text to the output file
                    file.write(extracted_text + '\n\n')  # Add some space between the content of different images
                    print(f"Processed and extracted text from {image_file}")
                    # Delete the PNG file after processing
                    os.remove(image_path)
                    break
                else:
                    # If no text was extracted, print a warning and retry once
                    print(f"No text extracted from {image_file} on attempt {attempt + 1}. Retrying..." if attempt == 0 else f"Retry failed for {image_file}. Continuing with script.")



def preprocess_image_for_ocr(image):
    """
    Preprocess the image for OCR by converting it to grayscale, enhancing contrast and sharpness,
    applying adaptive thresholding, and resizing the image.

    :param: image: PIL Image object to be preprocessed.

    :return: Preprocessed PIL Image object.
    """
    # Convert to grayscale to remove color noise
    image = image.convert('L')
    # Increase contrast to make text stand out
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # The value can be adjusted based on the image
    # Apply a sharpen filter to make the text clearer
    image = image.filter(ImageFilter.SHARPEN)
    # Apply adaptive thresholding to make the image binary
    image = image.point(lambda p: p > 128 and 255)
    # Optionally resize the image to make the text larger and clearer
    image = image.resize((image.width * 2, image.height * 2), Image.BICUBIC)
    return image


def get_output_file_name(default_name):
    """
    Prompt the user to enter a new file name if the default file name already exists.
    Allows overwriting the existing file by pressing Enter or specifying a new file name.

    :param: default_name: The default file name.

    :return: The selected file name (either the default or a new one).
    """
    while True:
        if not os.path.exists(default_name):
            return default_name
        default_name = input(f"The file '{default_name}' already exists. Enter a new file name or press Enter to overwrite: ")
        if default_name == "":
            return default_name


def clear_output_folder(output_folder):
    """
    Clear all PNG files from the specified output folder.

    :param: output_folder: Folder from which PNG files will be removed.

    :return: None
    """
    # Check for png files in the output folder and remove them
    for png_file in glob.glob(os.path.join(output_folder, '*.png')):
        os.remove(png_file)
    print(f"Cleared all PNG files from {output_folder}")


if __name__ == '__main__':
    pdf_path = 'pdf.pdf'  # Replace with your PDF file path
    output_folder = 'cropped_pages'  # Replace with your desired output directory
    header_height = 420  # Replace with the pixel height of the header to be removed
    footer_height = 420  # Replace with the pixel height of the footer to be removed
    column_offset = 15
    images_folder = 'cropped_pages'  # Replace with your actual folder containing images
    output_text_file = 'full_text.txt'  # The path to the output text file

    # Check if the default output file exists and get a new name if necessary
    output_text_file = get_output_file_name(output_text_file)

    # Ask user for the first and last page to process
    first_page = input("Enter the first page to process (default is 1): ")
    last_page = input("Enter the last page to process or press Enter to process until the end: ")

    # Convert user input to integers, using defaults if no input is provided
    first_page = int(first_page) if first_page.isdigit() else 1
    last_page = int(last_page) if last_page.isdigit() else None

    # Crop and extract text from PDF
    crop_page_to_columns(pdf_path, output_folder, header_height, footer_height, column_offset, first_page, last_page)
    extract_text_and_append_to_file(images_folder, output_text_file)