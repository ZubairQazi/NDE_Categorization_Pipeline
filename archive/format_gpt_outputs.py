import json
import logging

import pandas as pd

# Logging configuration
logging.basicConfig(level=logging.INFO)


def read_jsonl_file(file_path):
    """
    Reads a JSONL file and returns a list of JSON objects.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[dict]: List of JSON objects from the file.
    """
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    logging.info(f"Loaded {len(data)} records from {file_path}")
    return data


def parse_batch_output(batch_data):
    """
    Parses the output from the OpenAI batch request file.

    Args:
        batch_data (list): List of JSON objects from the batch output.

    Returns:
        pd.DataFrame: DataFrame containing custom_id, prompt, and response content.
    """
    records = []
    for entry in batch_data:
        custom_id = entry.get("custom_id")
        response_content = None
        status_code = entry.get("response", {}).get("status_code")

        # Only process successful responses
        if status_code == 200:
            response_content = (
                entry.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

        records.append(
            {
                "custom_id": custom_id,
                "status_code": status_code,
                "response_content": response_content,
            }
        )

    logging.info(f"Parsed {len(records)} entries from batch output.")
    return pd.DataFrame(records)


def save_to_csv(dataframe, output_path):
    """
    Saves the DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): DataFrame to save.
        output_path (str): Path to the output CSV file.
    """
    dataframe.to_csv(output_path, index=False)
    logging.info(f"Saved parsed output to {output_path}")


def main():
    input_file = input("Enter the path to the OpenAI batch output file (JSONL): ")
    output_csv = input("Enter the path for the output CSV file: ")

    batch_data = read_jsonl_file(input_file)
    parsed_data = parse_batch_output(batch_data)
    save_to_csv(parsed_data, output_csv)


if __name__ == "__main__":
    main()
