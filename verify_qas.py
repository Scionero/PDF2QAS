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

load_dotenv()

fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")

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
    question: str
    answer: str
    changed: str


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
    except Exception as e:
        logging.error(f"Error: {e}")
        user_input = input("User intervention is needed. Fix the issue and enter 'continue' to continue. If you "
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
    prompt_token_price = 0.0000004
    completion_token_price = 0.0000016
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
            retries = 0
            response = ""
            new_question = question
            new_answer = answer
            changed = ""

            while retries < max_retries:
                try:
                    response, usage = await verify_qa_async(f"Question: {question} \nAnswer: {answer}", system_content)
                    logging.info(f"Processing row {index}....")
                    # print(response)

                    # Parse the response and extract updated question and answer
                    response_df = parse_json_response_to_df(response)

                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                    tokens = usage.total_tokens
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_tokens += tokens

                    cost_for_row = prompt_tokens * prompt_token_price + completion_tokens * completion_token_price
                    costs_per_row.append(cost_for_row)

                    # Check if the DataFrame is not empty, indicating successful parsing
                    if not response_df.empty:
                        new_question = response_df.iloc[0]["question"]
                        new_answer = response_df.iloc[0]["answer"]
                        changed = response_df.iloc[0].get("changed", "None")
                        break  # Parsing successful, exit retry loop
                    else:
                        raise ValueError("Parsing Error")

                except Exception as e:
                    retries += 1
                    if retries < max_retries:
                        # Append the specified system message on retry due to parsing failure
                        system_content += ("The previous query failed, retry, and remember to follow all the rules. "
                                           "Your output must contain the enclosing brackets, quotes, and commas.")

            if retries == max_retries:
                error_message = f"Max retries reached while processing row {index}."
                logging.error(error_message)
                errors = [error_message]
                # Writing the failed entry to the rejects file
                reject_file.write(f"Original Question: {question}\n")
                reject_file.write(f"Original Answer: {answer}\n")
                reject_file.write("Errors Encountered:\n")
                for error in errors:
                    reject_file.write(f"- {error}\n")
                reject_file.write(f"Response Received:\n{response}\n")
                reject_file.write("-----\n\n")  # Delimiter for easy reading

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

            logging.info(f"Processed row {index}/{total_rows}. Estimated time remaining: {formatted_time_remaining}.")
            logging.info(f"Cost so far (May not be exact): ${cost_so_far:.4f}")
            logging.info(f"Estimated cost to finish: ${estimated_total_cost:.4f}")

    logging.info(f"Data saved to {output_path}")

    try:
        csv_to_jsonl(output_path, jsonl_file_path)
    except Exception as e:
        logging.error(f"Error converting CSV to JSONL: {e}")


if __name__ == '__main__':
    csv_path = "NEC2023_QA.csv"  # Replace with your CSV file path
    output_path = 'NEC2023_QA_Updated.csv'  # Replace with your output CSV file path
    jsonl_file_path = 'qas.jsonl'  # Replace with your JSONL file path
    asyncio.run(process_and_verify_pairs(csv_path, output_path, jsonl_file_path))
