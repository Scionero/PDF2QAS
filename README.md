
# PDF to Q&A's

## Overview

This document provides a guide for two Python scripts: one for cropping PDF pages into columns and extracting text using Optical Character Recognition (OCR) (PDF Processing script)
and another for processing text into question-answer pairs (Q&A script).

### PDF Processing Script

This script is designed for extracting information from PDF documents that are structured in a two-column layout. The script first converts each PDF page into a high-resolution image. It then crops these images into left and right columns, removing any predefined headers and footers. The cropped images are then processed using OCR to extract text. The extracted text is compiled and saved, providing a structured textual representation of the original PDF content.

**Modules Used:**
- `PyPDF2`: To read PDF files and get the number of pages.
- `pdf2image`: To convert PDF pages to images.
- `PIL (Pillow)`: For image manipulation like cropping.
- `pytesseract`: To apply OCR on images for text extraction.
- `os`, `glob`: For file and directory handling.

**Process Flow:**
1. Convert each page of a PDF to an image.
2. Crop each page into two columns based on predefined header, footer, and column offset values.
3. Apply OCR to extract text from each column.
4. Compile and save the extracted text.

### Q&A Verification Script

The Q&A Verification script is an advanced tool for processing large datasets of question-answer pairs. It employs an asynchronous approach to efficiently process each pair using the Fireworks AI API. The script reads each pair from a CSV file, verifies and potentially updates the content using AI, and then saves the revised pairs into new CSV and JSONL files. The script also includes error handling, logging, and cost estimation for API usage, making it a robust solution for large-scale data processing tasks.

**Modules Used:**
- `pandas`: To handle and manipulate large datasets.
- `asyncio`: For asynchronous programming, allowing concurrent processing.
- `logging`: For logging errors and information.
- `fireworks.client`: To interact with the Fireworks AI API for question-answer verification.
- `dotenv`: To manage environment variables for API keys.

**Process Flow:**
1. Read the dataset containing question-answer pairs.
2. Use the Fireworks AI API to verify and update each pair.
3. Log and handle errors and rate limits.
4. Save the updated pairs and generate cost estimates for the API usage.

## How to Run

### Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Scionero/PDF2QAS.git
   ```

2. **Environment File:**
   - Create and edit an `.env` file to include your specific API keys and other environment variables.

3. **Python Environment:**
   - Create a new virtual environment using Conda:
   ```bash
   conda create --name pdfqa python=3.10
   ```
   - Activate the environment:
   ```bash
   conda activate pdfqa
   ```

4. **Install Dependencies:**
   - Navigate to the directory containing `requirements.txt`.
   - Install the necessary Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration and Execution

1. **Customize Script Parameters:**
   - Set the path to your PDF file, define the output directory for the cropped images, and configure parameters for headers, footers, and column offset in the PDF Processing script.
   - Define paths for your input CSV file and the output files (CSV and JSONL) for the Q&A Verification script.

2. **Running the Scripts:**
   - Run the PDF Processing script:
   ```bash
   python text_to_qas.py
   ```
   - Then run the Q&A Verification script:
   ```bash
   python verify_qas.py
   ```

### Understanding Output Files

- **Cropped Images:** Generated by the PDF Processing script; split images of each PDF page.
- **Extracted Text File:** Contains the text extracted from the PDF.
- **Updated Q&A CSV:** The revised question-answer pairs in CSV format.
- **JSONL File:** A JSONL version of the updated question-answer pairs.

### Logs and Monitoring

- Scripts generate logs to monitor progress and troubleshoot issues. The Q&A Verification script also logs estimated API usage cost.
- Logs are saved in files named with the timestamp of the script's execution.

Follow these steps for a smooth execution of the scripts and effective processing of your documents. Be sure to check the logs for errors or important information.
