import os
import json
from datetime import time
from time import sleep

import openai
import pandas as pd
from dotenv import load_dotenv
import fireworks.client
from fireworks.client.error import RateLimitError
from pydantic import BaseModel, Field

load_dotenv()

fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")


class Result(BaseModel):
    question: str
    answer: str


def ai_query(query, context):
    system_content = """You are an expert test maker.  
                        - You can read any text file and create test questions and answers from it. 
                        - Try to come up with as many valid question/answer pairs as possible, within reason. 
                        - Only provide one question and one answer per pair, no multiple choice or True/False. 
                        - Verify that each question is unique, and that each answer is relevant to the question. 
                        - Answer in the form of a valid JSON object, with only two fields: 'question' and 'answer'. 
                        - Do not add any extra text, only the questions and answers.
                        - I will provide you with the previous chunk and response, as well as the new text for you to 
                        create Q/A pairs from. 
                        - Do not repeat any text from the previous response. 
                        - Use the previous chunk and response only as context to help in your Q/A creation. 
                        - There may be multiple spelling or grammar errors in the text due to the quality of the file, 
                        try and infer what the misspelled words might have been, and make sure your output doesn't
                        include any mistakes. 
                        -Some of the content was extracted from tables and diagrams, so some of it will be confusing 
                        and mostly meaningless, you can safely ignore it. 
                        - Make sure your response contains both a question and it's corresponding answer, 
                        I will treat the response as invalid if it doesn't.
                        - Do not include anything like 'The text does not contain' in your answers.
                            - If the text doesn't contain an answer to the question, don't bother providing the
                            question or answer.
                        """

    messages = [
        {"role": "system", "content": system_content},
        {"role": "assistant", "content": context},
        {"role": "user", "content": query},
    ]

    # If there is a previous response, add it as additional context
    if previous_response:
        messages.insert(1, {"role": "assistant", "content": previous_response})

    try:
        response = fireworks.client.chat.ChatCompletionV2.acreate(
            model="accounts/fireworks/models/mixtral-8x7b-instruct",
            response_format={"type": "json_object", "schema": json.dumps(Result.model_json_schema())},
            messages=messages,
            temperature=0.0,
            frequency_penalty=1.1,
            max_tokens=8000,
            stream=False,
            n=1
        )

        return response.choices[0].message.content, response.usage
    except RateLimitError as rate_limit_error:
        sleep(5)
    except Exception as e:
        print(f"Error: {e}")



def chunk_text(text, chunk_size=100):
    # Split the text into sentences
    sentences = text.split(". ")

    # Initialize variables to hold the chunks and the current chunk content
    chunks = []
    current_chunk = ""

    # Loop through sentences, creating chunks that are about the requested size
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            # If the sentence can fit into the current chunk, add it
            current_chunk += sentence + ". "
        else:
            # If the sentence can't fit, finish the current chunk and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    # Don't forget to add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def parse_json_response_to_df(json_response):
    try:
        questions = []
        answers = []
        lines = json_response.split('\n')

        for i, line in enumerate(lines):
            if '"question":' in line:
                # Extract the question text
                question = line.split('"question":', 1)[-1].strip(' ",')

                # Find the corresponding answer
                for j in range(i+1, len(lines)):
                    if '"answer":' in lines[j]:
                        answer = lines[j].split('"answer":', 1)[-1].strip(' ",')

                        # Add to lists
                        questions.append(question)
                        answers.append(answer)
                        break

        # Create DataFrame
        return pd.DataFrame({'question': questions, 'answer': answers})
    except Exception as e:
        print(f"Error parsing response: {e}")
        return pd.DataFrame(columns=['question', 'answer'])


def append_to_anomalies_file(chunk, file_path='anomalies.txt'):
    with open(file_path, 'a') as file:
        file.write(chunk + "\n\n")  # Append chunk with a couple of newlines for separation


def generate_qas(text_file_path):
    global previous_response

    # Read the content of the text file
    with open(text_file_path, 'r') as file:
        text_content = file.read()
    # Chunk the text
    text_chunks = chunk_text(text_content, chunk_size=200)
    previous_response = None
    master_df = pd.DataFrame(columns=['question', 'answer'])  # Master DataFrame to store all Q&A pairs
    chunk_counter = 0
    for chunk in text_chunks:
        try:
            print(f"Processing chunk {chunk_counter} of {len(text_chunks)}")
            output = ai_query(chunk, previous_response)
            print(output)  # Optional: for debugging purposes
            chunk_counter += 1
            previous_response = chunk + "\n\n" + output

            # Parse the response to DataFrame and append to the master DataFrame
            response_df = parse_json_response_to_df(output)
            master_df = pd.concat([master_df, response_df], ignore_index=True)
            print(f"Added {len(response_df)} Q&A pairs to the master DataFrame")
        except Exception as e:
            print(f"Error processing chunk: {e}")
            append_to_anomalies_file(chunk)  # Append problematic chunk to anomalies file
    # Write master DataFrame to CSV
    master_df.to_csv('NEC2023_QA.csv', index=False)


if __name__ == '__main__':
    text_file_path = 'NEC2023.txt'
    generate_qas(text_file_path)
