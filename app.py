import subprocess
import re
import pandas as pd
import streamlit as st
import logging
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
import json
import difflib
import diskcache as dc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up diskcache for caching
cache = dc.Cache(".llm_cache")

def initialize_llm():
    """Initializes the Google Vertex AI model."""
    logger.info("Initializing Google Vertex AI model.")
    try:
        llm = VertexAI(
            temperature=0.0,
            project="nexocode-lab",
            location="us-central1",
            model_name="gemini-1.5-flash-002",
        )
        logger.info("Successfully initialized Vertex AI model.")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI model: {e}")
        st.error(f"Error initializing LLM. Please check your environment variables: {e}")
        return None


proofread_template = PromptTemplate(
    input_variables=["chunk"],
    template="""Proofread the following mathematical text and correct any grammar mistakes: {chunk}.
        Ignore LaTeX formulas, figures and tables.

        List correction reasons for each correction in separate line.

    Return only the following as response:
    {{
        'needs_correction': true or false flag,
        'corrected_text': corrected_text,
        'correction_reason': correction_reason,
    }}"""
)


def convert_latex_to_markdown(latex_file):
    """Converts a LaTeX file to a Markdown file using pandoc, removing \ref{...} expressions."""
    logger.info("Converting LaTeX file to Markdown.")
    try:
        with open(latex_file, "r") as f:
            latex_content = f.read()
        cleaned_content = re.sub(r"\\ref\{.*?\}", "", latex_content)

        temp_file = "temp_latex_file.tex"
        with open(temp_file, "w") as f:
            f.write(cleaned_content)

        output_file = "output.md"
        subprocess.run(["pandoc", temp_file, "-o", output_file], check=True)

        logger.info("Successfully converted LaTeX to Markdown.")
        with open(output_file, "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to convert LaTeX to Markdown: {e}")
        return ""


def split_text_into_chunks(markdown_text):
    """Splits the Markdown text into chunks by empty lines and concatenates short chunks."""
    logger.info("Splitting Markdown text into chunks by empty lines.")
    chunks = [chunk.strip() for chunk in re.split(r'(?:\n\s*\n)+', markdown_text) if chunk.strip()]

    concatenated_chunks = []
    temp_chunk = ""

    for chunk in chunks:
        if temp_chunk:
            temp_chunk += "\n" + chunk
            concatenated_chunks.append(temp_chunk)
            temp_chunk = ""
        elif len(chunk.splitlines()) <= 2:
            temp_chunk = chunk
        else:
            concatenated_chunks.append(chunk)

    if temp_chunk:
        concatenated_chunks.append(temp_chunk)

    return concatenated_chunks


def highlight_differences(original, corrected):
    """Highlights differences between original and corrected text using difflib, ignoring whitespaces and matching whole words."""
    original_words = original.split()
    corrected_words = corrected.split()

    diff = difflib.SequenceMatcher(None, original_words, corrected_words)
    highlighted_lines = []

    for tag, i1, i2, j1, j2 in diff.get_opcodes():
        if tag == 'replace':
            highlighted_lines.append((" ".join(original_words[i1:i2]), " ".join(corrected_words[j1:j2]), True))
        elif tag == 'equal':
            highlighted_lines.append((" ".join(original_words[i1:i2]), " ".join(corrected_words[j1:j2]), False))
    return highlighted_lines


def proofread_chunk(chunk, llm):
    """Uses LLM to proofread a chunk of text with caching."""
    cache_key = f"proofread_{hash(chunk)}"
    if cache_key in cache:
        logger.info("Cache hit for chunk.")
        return cache[cache_key]

    prompt = proofread_template.format(chunk=chunk)
    response = llm.invoke(prompt)
    cache[cache_key] = response
    logger.info("Response cached.")
    return response


def replace_latex_math_with_equation(text):
    """Replace LaTeX math formulas in a string with the word 'EQUATION'."""
    return re.sub(r'\$\$(.*?)\$\$', 'equation placeholder', text, flags=re.DOTALL)


def main():
    st.set_page_config(layout="wide")
    llm = initialize_llm()

    st.title("LaTeX to Markdown Converter with Grammar Check")

    uploaded_file = st.file_uploader("Upload a LaTeX file", type="tex")
    if uploaded_file:
        with open("uploaded.tex", "wb") as f:
            f.write(uploaded_file.read())

        markdown_text = convert_latex_to_markdown("uploaded.tex")

        if markdown_text:
            chunks = split_text_into_chunks(markdown_text)[:2]

            if "accepted_text" not in st.session_state:
                st.session_state["accepted_text"] = []

            total_chunks = len(chunks)
            progress_bar = st.progress(0)

            for i, chunk in enumerate(chunks):
                chunk = replace_latex_math_with_equation(chunk)

                if llm:
                    try:
                        response = proofread_chunk(chunk, llm)
                        json_string = response[response.find("{"):response.rfind("}") + 1]
                        response_json = json.loads(json_string)
                    except Exception as e:
                        logger.error(f"Failed to process chunk: {chunk} | Error: {e}")
                        st.error(f"Failed to process chunk: {e}")
                        continue

                    if response_json.get("needs_correction"):
                        original_text = chunk
                        corrected_text = response_json.get("corrected_text", "")

                        highlighted_lines = highlight_differences(original_text, corrected_text)

                        was_last_one_ok = True
                        fine_text_to_append = ""

                        for j, (line_original, line_corrected, is_different) in enumerate(highlighted_lines):

                            if was_last_one_ok is True and is_different is True:
                                st.markdown(fine_text_to_append)

                                fine_text_to_append = ""
                                was_last_one_ok = False

                            if was_last_one_ok is True and is_different is False:
                                st.session_state.accepted_text.append(line_original)
                                fine_text_to_append += " " + line_original

                            if was_last_one_ok is False and is_different is False:
                                was_last_one_ok = True
                                fine_text_to_append = line_original

                            if was_last_one_ok is False and is_different is True:
                                st.session_state.accepted_text.append("PLACEHOLDER")
                                st.markdown(f"<div><span style='color: red;'>{line_original}</span> → <span style='color: green;'>{line_corrected}</span></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(chunk)
                        st.session_state.accepted_text.append(chunk)

                progress_bar.progress((i + 1) / total_chunks)

            # if st.button("Export Accepted Suggestions"):
            #     df = pd.DataFrame(results)
            #     df.to_excel("accepted_suggestions.xlsx", index=False)
            #     st.success("Exported suggestions to accepted_suggestions.xlsx.")
            #
            # if st.button("Download Full Text as Markdown"):
            #     full_text = "\n\n".join(st.session_state.accepted_text)
            #     with open("final_text.md", "w") as f:
            #         f.write(full_text)
            #     st.download_button(
            #         label="Download Markdown File",
            #         data=full_text,
            #         file_name="final_text.md",
            #         mime="text/markdown"
            #     )

if __name__ == "__main__":
    main()
