import json
import os
import argparse
import re
import random
import numpy as np
import torch
import string

"""
This code follows lm-harness nq-open answer extraction and metrics
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/nq_open/nq_open.yaml
"""

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

                        
def exact_match_hf_evaluate(
    prediction,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            prediction = np.array([re.sub(s, "", x) for x in prediction])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        prediction = np.asarray(prediction)
        references = np.asarray(references)

    if ignore_case:
        prediction = np.char.lower(prediction)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        prediction = np.char.translate(prediction, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        prediction = np.char.translate(prediction, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = []
    for reference in references:
        if prediction[0] in reference or reference in prediction[0]:
            score_list.append(1)
            break
        
    print("prediction", prediction)
    print("references", references)
    print("score_list", score_list)
    
    return {"exact_match": np.mean(score_list)}


def is_exact_match(response, question, answers, answer_label):
    """
    Check if the normalized response contains an exact match of any normalized answer immediately following the question.
    """
    
    try:
        start_index = response.index(question) + len(question)
        subsequent_text = response[start_index:]
        subsequent_text = subsequent_text.split('\n', 1)[0] # stop at the first complete line
        subsequent_text = subsequent_text.strip().replace(f'{answer_label}:', '')

        score = exact_match_hf_evaluate(references=answers, 
                                        prediction=[subsequent_text], 
                                        regexes_to_ignore= ["\\b(?:The |the |An |A |The |a |an )"], 
                                        ignore_case=True, 
                                        ignore_punctuation=True)
        
        return 1 if score['exact_match'] > 0 else 0        
    except ValueError:
        return 0  # Question not found in response


def evaluate_dataset(data, answer_label):
    """
    Evaluates the dataset for exact matches between provided answers and responses after normalization.
    """
    labeled_data = []
    true_count = 0
    false_count = 0

    for entry in data:
        question = entry['question']

        answers = entry['answer'] if isinstance(entry['answer'], list) else [entry['answer']]
        response = entry['response']

        label = is_exact_match(response, question, answers, answer_label)
        
        if label == 1:
            true_count += 1
        else:
            false_count += 1

        labeled_data.append({
            'question': question,
            'answer': answers,
            'response': response,
            'label': label
        })

    return labeled_data, true_count, false_count


def process_answers(file_path, answer_label):
    """
    Processes each JSON file in the specified directory, evaluating each for exact match correctness.
    Saves the labeled data back to new files annotated with labeling results.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    labeled_data, true_count, false_count = evaluate_dataset(data, answer_label)

    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)

    labeled_filename = f"{name}_labeled_true_{true_count}_false_{false_count}{ext}"

    # Create new file path
    labeled_file_path = os.path.join(directory, labeled_filename)

    # Save the labeled data to a new file
    with open(labeled_file_path, 'w') as file:
        json.dump(labeled_data, file, indent=4)

    print(f"Directory: {directory}")
    print(f"Processed File: {filename}")
    print(f"Labeled File: {labeled_filename}")
    print(f"True count: {true_count}")
    print(f"False count: {false_count}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a JSON file for exact match answers following lm-harness.")
    parser.add_argument('file_path', type=str, help="Path to the file containing the model responses.")
    parser.add_argument('--answer_label', type=str, required=False, default="A", help="answer label")

    args = parser.parse_args()

    process_answers(args.file_path, args.answer_label)


if __name__ == "__main__":
    main()
