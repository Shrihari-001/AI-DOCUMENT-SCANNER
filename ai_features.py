
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "Gemini API key .env file mein nahi mili!"
    )

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.5-flash-lite"

# FEATURE 1: TEST GEMINI
def test_gemini():
    response = client.models.generate_content(
        model=MODEL,
        contents="Say hello in simple English."
    )
    return response.text

# FEATURE 2: AI SUMMARY
def generate_summary(text):
    prompt = f"""
    Read the document carefully.
    Create a clear and short summary.
    Include main topics and important information.
    Use simple English.

    Document:
    {text}
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text

# FEATURE 3: EASY STUDY NOTES
def generate_easy_notes(text):
    prompt = f"""
    Convert this document into easy study notes.

    Requirements:
    1. Use simple English.
    2. Use clear headings.
    3. Explain important points.
    4. Use bullet points.
    5. Keep the original meaning.

    Document:
    {text}
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text

# FEATURE 4: AI QUIZ
def generate_quiz(text):
    prompt = f"""
    Create 10 multiple choice questions from this document.

    Requirements:
    1. Give 4 options: A, B, C and D.
    2. Give only one correct answer.
    3. Show the correct answer.
    4. Give a short explanation.
    5. Use simple English.

    Document:
    {text}
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text

# FEATURE 5: AI FLASHCARDS
def generate_flashcards(text):
    prompt = f"""
    Create 10 study flashcards from this document.

    Each flashcard must contain:
    Question: A clear question.
    Answer: A short and useful answer.

    Use simple English.
    Cover important topics.
    Avoid repeated questions.

    Document:
    {text}
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text

# FEATURE 6: AI PPT SLIDE CONTENT
def generate_ppt_content(text):

    prompt = f"""
    Create a PowerPoint presentation from this document.

    Return ONLY valid JSON in this format:

    {{
        "title": "Presentation Title",
        "slides": [
            {{
                "title": "Introduction",
                "points": [
                    "Point one",
                    "Point two",
                    "Point three"
                ]
            }}
        ]
    }}

    Requirements:
    1. Create 6 to 8 slides.
    2. Include an introduction.
    3. Cover important topics.
    4. Use simple English.
    5. Include a conclusion slide.
    6. Each slide should have 3 to 5 short points.
    7. Do not include markdown code fences.

    Document:
    {text}
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    result = response.text.strip()

    # Remove possible markdown code fences
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
        result = result.rsplit("```", 1)[0]

    return json.loads(result)