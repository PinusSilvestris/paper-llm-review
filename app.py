import os
import logging
import re
import streamlit as st
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from typing import List

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize LLM
def initialize_llm():
    """Initializes the Google Vertex AI model."""
    logger.info("Initializing Google Vertex AI model.")
    try:
        llm = VertexAI(
            temperature=0.0,
            project=os.environ.get("VERTEX_PROJECT"),
            location=os.environ.get("VERTEX_LOCATION"),
            model_name=os.environ.get("VERTEX_MODEL", "gemini-1.5-flash-002")
        )
        logger.info("Successfully initialized Vertex AI model.")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI model: {e}")
        st.error(f"Error initializing LLM. Please check your environment variables: {e}")
        return None

llm = initialize_llm()

def get_grammar_check_pipeline():
    """
    Creates and returns a LangChain LLMChain configured for grammar checking.
    """
    logger.debug("Creating grammar check pipeline.")
    prompt = PromptTemplate(
        template="""
        You are an expert English grammar checker. Given the following paragraph,
        identify any grammatical errors and provide a suggestion for each correction.
        Do not change the original text but provide the suggestion next to it, so the user can review them and accept them, modify or reject.
        If there are no errors, do not include suggestions. The text is in latex format and you should keep it in the same format.
        Paragraph:
        {paragraph}
        """,
        input_variables=["paragraph"]
    )
    chain = prompt | llm
    return chain

def call_llm_grammar_check(paragraph: str) -> str:
    """
    Uses the LLM chain to perform grammar checks on the provided paragraph.
    """
    if not llm:
        st.error("LLM not initialized. Please check the logs.")
        return ""
    logger.debug("Preparing to call LLM chain for grammar check.")
    chain = get_grammar_check_pipeline()
    try:
        result = chain.invoke({"paragraph": paragraph})
        logger.debug("Raw LLM output for grammar check: %s", result)
        return result
    except Exception as e:
        logger.error(f"Error during LLM call: {e}")
        st.error("Error during LLM call. Check logs for details.")
        return ""

def parse_latex_file(file_path: str) -> List[str]:
    """
    Reads a LaTeX file, extracts paragraph content, and keeps the latex format.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        # Split the content into paragraphs, keeping latex commands
        # This regex will not split inside of a single paragraph with math commands
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        logger.info(f"Successfully parsed LaTeX file with {len(paragraphs)} paragraphs.")
        return paragraphs
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        st.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading or parsing the LaTeX file: {e}")
        st.error(f"Error reading or parsing the LaTeX file: {e}")
        return []


def apply_changes(
        paragraphs: List[str],
        suggestions: List[str],
        user_edits: List[str],
        actions: List[str]
) -> List[str]:
    """Applies the user accepted or modified changes to the original text."""
    updated_paragraphs = []

    for original, suggestion, user_edit, action in zip(paragraphs, suggestions, user_edits, actions):
        if action == "accept":
            if suggestion:
                updated_paragraphs.append(suggestion)
            else:
                updated_paragraphs.append(original)
        elif action == "reject":
            updated_paragraphs.append(original)
        elif action == "modify":
            updated_paragraphs.append(user_edit)
        else:  # No action
            updated_paragraphs.append(original)
    return updated_paragraphs

def main():
    st.title("LaTeX Grammar Checker")

    # File uploader
    uploaded_file = st.file_uploader("Upload a LaTeX file", type=["tex", "latex"])
    if uploaded_file:
        # Save the file temporarily for processing
        file_path = "temp_file.tex"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        paragraphs = parse_latex_file(file_path)

        if not paragraphs:
            st.error("No paragraphs found in file.")
            return

        if 'suggestions' not in st.session_state:
            st.session_state['suggestions'] = [""] * len(paragraphs)
        if 'user_edits' not in st.session_state:
            st.session_state['user_edits'] = [""] * len(paragraphs)
        if 'actions' not in st.session_state:
            st.session_state['actions'] = [""] * len(paragraphs)



        # Process each paragraph
        for i, paragraph in enumerate(paragraphs):
            st.subheader(f"Paragraph {i+1}")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original:**")
                st.write(paragraph)

            with col2:
                st.markdown("**Proposed:**")
                if not st.session_state['suggestions'][i]: # Check to avoid multiple calls
                    suggestion = call_llm_grammar_check(paragraph)
                    st.session_state['suggestions'][i] = suggestion
                st.write(st.session_state['suggestions'][i])

                st.session_state['user_edits'][i] = st.text_area("Edit", value = st.session_state['suggestions'][i], key = f"edit_{i}")

                col_buttons = st.columns(3)
                with col_buttons[0]:
                    if st.button("Accept", key = f"accept_{i}"):
                        st.session_state['actions'][i] = "accept"
                with col_buttons[1]:
                    if st.button("Reject", key = f"reject_{i}"):
                        st.session_state['actions'][i] = "reject"
                with col_buttons[2]:
                    if st.button("Modify", key = f"modify_{i}"):
                        st.session_state['actions'][i] = "modify"

        if st.button("Apply Changes"):
            updated_paragraphs = apply_changes(
                paragraphs,
                st.session_state['suggestions'],
                st.session_state['user_edits'],
                st.session_state['actions']
            )
            st.session_state['actions'] = [""] * len(paragraphs)
            st.session_state['suggestions'] = [""] * len(paragraphs)

            st.session_state['user_edits'] = [""] * len(paragraphs)

            st.write("## Modified Document")
            st.write("\n\n".join(updated_paragraphs))

if __name__ == "__main__":
    main()