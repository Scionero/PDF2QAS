import functools
import os
import csv
import json
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import fireworks.client
from fireworks.client.error import (RateLimitError, PermissionError, InvalidRequestError, AuthenticationError,
                                    InternalServerError, ServiceUnavailableError)
import asyncio
import logging
from pydantic import BaseModel, Field
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


load_dotenv()

fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")
app_password = os.getenv("GMAIL_APP_PASSKEY")
email_address = os.getenv("GMAIL_ADDRESS")

# Create a timestamp for the log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'error_log_{timestamp}.log'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create a file handler for logging error messages
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add the file handler to the root logger
logging.getLogger('').addHandler(file_handler)


class Result(BaseModel):
    question: str = Field(..., title="Question")
    answer: str = Field(..., title="Answer")
    changed: str = Field("None", title="Changed")


class MaxRetriesExceededError(Exception):
    def __init__(self, question, answer, message="Max retries reached"):
        self.question = question
        self.answer = answer
        self.message = message
        super().__init__(self.message)



async def verify_qa_async(query, system_content):
    """
    Asynchronously verifies a question-answer pair using a chat completion model.

    :param: query: The user's question.
    :param: system_content: The system's response.

    :return: The generated answer.
    :return: The total tokens used for this query.

    :raises: RateLimitError: If the rate limit is exceeded.
    :raises:Exception: If an error occurs during the verification process.
    """

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]

    try:
        response = await fireworks.client.chat.ChatCompletionV2.acreate(
            model="accounts/fireworks/models/mixtral-8x7b-instruct",
            response_format={"type": "json_object", "schema": json.dumps(Result.model_json_schema())},
            messages=messages,
            temperature=0.0,
            frequency_penalty=1.1,
            max_tokens=2000,
            stream=False,
            n=1
        )

        return response.choices[0].message.content, response.usage

    except RateLimitError as rate_limit_error:
        logging.error(f"Rate limit error: {rate_limit_error}")
        await asyncio.sleep(5)
    except ServiceUnavailableError as service_unavailable_error:
        logging.error(f"Service unavailable error: {service_unavailable_error}")
        await asyncio.sleep(5)
    except Exception as e:
        logging.error(f"Error: {e}")
        send_email("An error has occurred that stopped the script from running. User intervention is required before "
                   "the script can continue.", f"Error: {e}", log_filename)
        user_input = input("User intervention is needed. Fix the issue and press enter to continue. If you "
                           "would like to exit, enter 'exit'.").lower()
        if user_input == "exit":
            raise SystemExit


def parse_json_response_to_df(json_response):
    try:
        # Convert the JSON string to a Python dictionary
        data_dict = json.loads(json_response)

        # Wrap the dictionary in a list and create a DataFrame
        df = pd.DataFrame([data_dict])
        return df
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}")
        return pd.DataFrame(columns=["question", "answer", "changed"])
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        return pd.DataFrame(columns=["question", "answer", "changed"])


def csv_to_jsonl(csv_file_path, jsonl_file_path):
    """
    Convert a CSV file to a JSONL file.

    :param: csv_file_path: The path to the CSV file.
    :param: jsonl_file_path: The path to the JSONL file.

    :return: None
    """
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)

        with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
            for row in reader:
                # Formatting the row as required
                formatted_row = {"text": f"<Question>: {row['question']} <Answer>: {row['answer']}"}

                # Writing the formatted row to the JSONL file
                jsonl_file.write(json.dumps(formatted_row) + "\n")


def format_seconds_to_hms(seconds):
    """
    Converts seconds to hours, minutes, and seconds format. HH:MM:SS

    :param: seconds: The number of seconds.

    :return: The formatted time in hours, minutes, and seconds.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def send_email(subject, body, attachment_path=None):
    try:
        # Set up the email server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        # Login to the email account
        server.login(email_address, app_password)

        # Create the email
        email = MIMEMultipart()
        email['From'] = email_address
        email['To'] = email_address
        email['Subject'] = subject

        # Attach the email body
        email.attach(MIMEText(body, 'plain'))

        # Check if there is an attachment
        if attachment_path and os.path.isfile(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(attachment_path)}")
            email.attach(part)

        # Send the email
        server.send_message(email)
        server.quit()
    except Exception as e:
        logging.error(f"Error sending email: {e}")


def count_errors_in_log(log_file):
    """
    Counts the number of error entries in the log file.
    :param log_file: Path to the log file.
    :return: The number of error entries.
    """
    error_count = 0
    try:
        with open(log_file, 'r') as file:
            for line in file:
                if 'ERROR' in line:  # Assuming ERROR is logged for each error
                    error_count += 1
    except Exception as e:
        logging.error(f"Error reading log file: {e}")
        # Optional: Handle this exception based on your needs.
    return error_count


def token_time_calculator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        response, response_df, usage = await func(*args, **kwargs)  # Corrected to expect three values
        end_time = time.time()

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        tokens = usage.total_tokens

        prompt_token_price = 0.0000004
        completion_token_price = 0.0000016
        cost_for_row = prompt_tokens * prompt_token_price + completion_tokens * completion_token_price

        time_taken = end_time - start_time
        formatted_time = format_seconds_to_hms(time_taken)

        return response, response_df, usage, cost_for_row, formatted_time, tokens  # Return all the necessary values
    return wrapper


def check_error_threshold_and_exit(log_file, threshold=100):
    """
    Checks if the number of errors in the log file exceeds the threshold and exits if it does.
    :param log_file: Path to the log file.
    :param threshold: The maximum number of errors allowed.
    """
    if count_errors_in_log(log_file) > threshold:
        logging.error("Error threshold exceeded. Exiting script.")
        exit(1)


@token_time_calculator
async def process_qa_pair_with_retries(question, answer, system_content, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            response, usage = await verify_qa_async(f"Question: {question} \nAnswer: {answer}", system_content)
            response_df = parse_json_response_to_df(response)

            # Check if the DataFrame is not empty, indicating successful parsing
            if not response_df.empty:
                return response, response_df, usage

            raise ValueError("Parsing Error")

        except Exception as e:
            retries += 1
            logging.error(f"Retry {retries}/{max_retries} for QA pair due to error: {e}")
            if retries >= max_retries:
                logging.error(f"Max retries reached for QA pair: {e}")
                check_error_threshold_and_exit(log_filename)
                raise MaxRetriesExceededError(question, answer, f"Max retries reached. Error: {e}")

            # Modify system_content for retry if needed
            system_content += (" The previous query failed, retry, and remember to follow all the rules. "
                               "Your output must contain the enclosing brackets, quotes, and commas.")

            # Optionally, include a delay between retries
            await asyncio.sleep(1)  # Delay for 1 second (or as appropriate)


async def reject_handler(answer, index, question, reject_file, response):
    error_message = f"Max retries reached while processing row {index}."
    logging.error(error_message)
    check_error_threshold_and_exit(log_filename)
    errors = [error_message]
    # Writing the failed entry to the rejects file
    reject_file.write(f"Original Question: {question}\n")
    reject_file.write(f"Original Answer: {answer}\n")
    reject_file.write("Errors Encountered:\n")
    for error in errors:
        reject_file.write(f"- {error}\n")
    reject_file.write(f"Response Received:\n{response}\n")
    reject_file.write("-----\n\n")  # Delimiter for easy reading


async def process_and_verify_pairs(csv_path, output_path, jsonl_file_path):
    """
    Main function to process a CSV file containing question/answer pairs, verify them using an AI model, and save the updated pairs to a new CSV file.

    The function reads a CSV file containing question/answer pairs, and for each pair:
    - Verifies the pair by calling the `verify_qa` function with the question and answer.
    - Parses the JSON response to extract the updated question, answer, and changes.
    - Writes the updated pair to a new CSV file.
    - If there are parsing errors or the maximum number of retries is reached, the pair is written to a rejects file.

    :param: csv_path: The path to the CSV file containing question/answer pairs.
    :param: output_path: The path to the new CSV file containing the updated question/answer pairs.
    :param: jsonl_file_path: The path to the JSONL file containing the original question/answer pairs.

    :return: None
    """

    # Check and delete output files if they exist
    for file_path in [output_path, jsonl_file_path, "rejects.txt"]:
        if os.path.exists(file_path):
            os.remove(file_path)

    system_content = """You are the assistant to a professor who is tasked with creating question/answer pairs from 
                            a given text. 
                            - Your task is to verify if the question and answer pair are correct.
                            - If the question or answer are incorrect, malformed, or incomplete, you should correct it.
                            - Some of the content will contain information that supersedes your knowledge, do not change
                            any values in the content.
                            - If needed, restructure the question and answer pair.
                            - Add a third line to the content, called "changed", to indicate what was changed.
                            - Fill in the "changed" field with whatever changes were made to the question or answer, 
                            if nothing was changed, fill in "None".
                            - Your output should consist of five lines only, in proper JSON, with the following 
                            format:
                            {
                                "question": "The question",
                                "answer": "The answer",
                                "changed": "What was changed"
                            },
                            - Do not include any text before or after the JSON object.
                            - Only five lines per output. 
    """
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    start_time = time.time()

    max_retries = 3  # Set a maximum number of retries
    reject_file = "rejects.txt"

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    costs_per_row = []
    cost_so_far = 0

    # Open the output file in write mode
    with (open(output_path, mode='w', newline='', encoding='utf-8') as file, open(reject_file, mode='a',
                                                                                     encoding='utf-8') as reject_file):
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['question', 'answer', 'changed'])

        for index, (idx, row) in enumerate(df.iterrows(), start=1):
            question = row["question"]
            answer = row["answer"]

            try:
                response, response_df, usage, cost_for_row, formatted_time, tokens = (
                    await process_qa_pair_with_retries(question, answer, system_content, max_retries
                    ))

                # Extract and write updated row to CSV
                new_question = response_df.iloc[0]["question"]
                new_answer = response_df.iloc[0]["answer"]
                changed = response_df.iloc[0].get("changed")
                writer.writerow([new_question, new_answer, changed])

                # Accumulate total costs and tokens
                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens
                total_tokens += tokens
                costs_per_row.append(cost_for_row)

            except MaxRetriesExceededError as max_retries_error:
                await reject_handler(answer, index, question, reject_file, response)

            # Write the updated row to the CSV file
            writer.writerow([new_question, new_answer, changed])

            iteration_end = time.time()
            total_time_taken = iteration_end - start_time
            average_time_per_row = total_time_taken / index
            estimated_total_time = average_time_per_row * total_rows
            estimated_time_remaining = estimated_total_time - total_time_taken

            formatted_time_remaining = format_seconds_to_hms(estimated_time_remaining)

            average_cost_per_row = sum(costs_per_row) / len(costs_per_row)
            estimated_total_cost = average_cost_per_row * total_rows
            cost_so_far += cost_for_row

            logging.info(f"Processed row {index}/{total_rows}.\n"
                         f"Estimated time remaining: {formatted_time_remaining}.\n"
                         f"Cost so far (May not be exact): ${cost_so_far:.4f}\n"
                         f"Estimated cost to finish: ${estimated_total_cost:.4f}\n")

    logging.info(f"Data saved to {output_path}")
    total_errors = count_errors_in_log(log_filename)

    send_email(subject="Verify Q/A's script finished successfully",
               body=(f"Total time taken: {format_seconds_to_hms(total_time_taken)}.\n"
                     f"Total calculated cost: {cost_so_far:.4f}.\n"
                     f"Total number of errors encountered: {total_errors}.\n"
                     f"Error log is attached.\n"),
               attachment_path=log_filename
               )

    try:
        csv_to_jsonl(output_path, jsonl_file_path)
    except Exception as e:
        logging.error(f"Error converting CSV to JSONL: {e}")


if __name__ == '__main__':
    csv_path = "NEC2023_QA.csv"  # Replace with your CSV file path
    output_path = 'NEC2023_QA_Updated.csv'  # Replace with your output CSV file path
    jsonl_file_path = 'qas.jsonl'  # Replace with your JSONL file path
    asyncio.run(process_and_verify_pairs(csv_path, output_path, jsonl_file_path))
