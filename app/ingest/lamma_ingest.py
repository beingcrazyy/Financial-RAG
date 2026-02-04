import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

load_dotenv()

# 1. Configuration - "Telling the AI what to look for"
# This instruction is sent to the parsing model to improve extraction quality.
finance_instructions = """
This document is a formal SEC 10-K filing. 
1. Tables are critical: ensure all financial tables are converted into clean Markdown format.
2. Do not merge columns: keep 'Year 2023' and 'Year 2022' columns distinct.
3. Preserve row headers: labels like 'Net Income' must stay aligned with their values.
"""

parser = LlamaParse(
    api_key= os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",       # Options: 'text', 'markdown', 'json'
    num_workers=4,                # Parallel processing for 100+ page 10-Ks
    parsing_instruction=finance_instructions,
    verbose=True
)

# 2. Layout-Aware Loading
# We map the '.pdf' extension to our high-powered parser.
file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader(
    input_dir="./doc", 
    file_extractor=file_extractor
).load_data()

# 3. Verification: Print the first table found to see the difference
print(f"Parsed {len(documents)} pages.")
print("Sample of structured data:")
print(documents[0].text[:1000]) # You will see '| Column 1 | Column 2 |' structure here.