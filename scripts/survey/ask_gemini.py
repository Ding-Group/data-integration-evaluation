"""
Read a PDF and ask Gemini 2.5 about dataset characteristics.

Usage:
    ask_gemini.py <file>

"""

import json
import os

from docopt import docopt
from pypdf import PdfReader

from google import genai
from google.genai import types


def extract_pdf_text(path):
    reader = PdfReader(path)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    return "\n".join(all_text)


def main():
    args = docopt(__doc__)
    file = args["<file>"]

    client = genai.Client()

    pdf_text = extract_pdf_text(file)
    print(len(pdf_text))
    print(len(pdf_text.split()))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="""
You are assisting a researcher in extracting information from the main text of academic papers.
The papers all concern single-cell transcriptomics data integration methods.
Return all answers strictly as a JSON dictionary with the following keys:
- "question1"
- "question2"
- "question3"
Answer each question bullet points where each key of the JSON is a list. Do not elaborate upon the findings.
If you cannot find the answer to a question in the text given, set the value to "not found".
The answers may be hidden in supplementary files that are not included.
"""
        ),
        contents=f"""The following text is extracted from an academic paper:\n\n{pdf_text}

Question 1: What metrics do the authors use to benchmark their method against existing methods for data integration/batch correction? Place each metric in its own bullet point.
Question 2: What datasets do they use to demonstrate strong data integration/batch correction performance? For each dataset, refer to the citation and output the first-author, year of publication, and journal the dataset was published in.
Question 3: Which existing methods do the authors benchmark against for data integration/batch correction?"""
    )
    print(response.text)
    text = response.text
    text = text.lstrip('```json')
    text = text.rstrip('```')
    name = os.path.splitext(os.path.basename(file))[0]
    with open(f"/ubc/cs/research/beaver/projects/Single-cell-integration-metrics/reports/survey/{name}_gemini.json", 'w') as f:
        f.write(text)


if __name__ == "__main__":
    main()
