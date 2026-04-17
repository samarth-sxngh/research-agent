import logging
import json
import requests
import os
from typing import List, Dict, Any
from dataclasses import dataclass

from src.document_processing.doc_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")


@dataclass
class PodcastScript:
    script: List[Dict[str, str]]
    source_document: str
    total_lines: int
    estimated_duration: str

    def get_speaker_lines(self, speaker: str) -> List[str]:
        return [item[speaker] for item in self.script if speaker in item]

    def to_json(self) -> str:
        return json.dumps({
            "script": self.script,
            "metadata": {
                "source_document": self.source_document,
                "total_lines": self.total_lines,
                "estimated_duration": self.estimated_duration
            }
        }, indent=2)


class PodcastScriptGenerator:
    def __init__(self, openai_api_key: str = None, model_name: str = None):
        # openai_api_key and model_name kept for signature compat — unused
        self.model = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL
        self.doc_processor = DocumentProcessor()
        logger.info(f"PodcastScriptGenerator initialized with Ollama model: {self.model}")

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]

    def generate_script_from_document(
        self,
        document_path: str,
        podcast_style: str = "conversational",
        target_duration: str = "10 minutes"
    ) -> PodcastScript:
        logger.info(f"Generating podcast script from: {document_path}")
        chunks = self.doc_processor.process_document(document_path)
        if not chunks:
            raise ValueError("No content extracted from document")
        document_content = "\n\n".join([chunk.content for chunk in chunks])
        source_name = chunks[0].source_file
        script_data = self._generate_conversation_script(document_content, podcast_style, target_duration)
        return PodcastScript(
            script=script_data["script"],
            source_document=source_name,
            total_lines=len(script_data["script"]),
            estimated_duration=target_duration
        )

    def generate_script_from_text(
        self,
        text_content: str,
        source_name: str = "Text Input",
        podcast_style: str = "conversational",
        target_duration: str = "10 minutes"
    ) -> PodcastScript:
        logger.info("Generating podcast script from text input")
        script_data = self._generate_conversation_script(text_content, podcast_style, target_duration)
        return PodcastScript(
            script=script_data["script"],
            source_document=source_name,
            total_lines=len(script_data["script"]),
            estimated_duration=target_duration
        )

    def generate_script_from_website(
        self,
        website_chunks: List[Any],
        source_url: str,
        podcast_style: str = "conversational",
        target_duration: str = "10 minutes"
    ) -> PodcastScript:
        logger.info(f"Generating podcast script from website: {source_url}")
        if not website_chunks:
            raise ValueError("No website content provided")
        website_content = "\n\n".join([chunk.content for chunk in website_chunks])
        script_data = self._generate_conversation_script(website_content, podcast_style, target_duration)
        return PodcastScript(
            script=script_data["script"],
            source_document=source_url,
            total_lines=len(script_data["script"]),
            estimated_duration=target_duration
        )

    def _generate_conversation_script(
        self,
        document_content: str,
        podcast_style: str,
        target_duration: str
    ) -> Dict[str, Any]:
        style_prompts = {
            "conversational": "friendly and natural",
            "educational": "one explains, other asks questions",
            "interview": "Speaker 1 interviews Speaker 2",
            "debate": "different perspectives respectfully"
        }
        style = style_prompts.get(podcast_style, "friendly and natural")
        doc_snippet = document_content[:2000]

        prompt = f"""Write a podcast conversation between Speaker 1 and Speaker 2 about the document below.
Style: {style}
Format each line exactly like this:
Speaker 1: [dialogue here]
Speaker 2: [dialogue here]
Speaker 1: [dialogue here]

Write 8-12 alternating lines. Start with an introduction and end with a wrap-up.

Document:
{doc_snippet}

Podcast script:
Speaker 1:"""

        try:
            raw = self._call_ollama(prompt)
            # Prepend what the prompt already started
            raw = "Speaker 1:" + raw
            script = self._extract_script_from_text(raw)
            if not script or len(script) < 2:
                raise ValueError(f"Could not extract script from response. Got {len(script)} lines.")
            return {"script": script}
        except Exception as e:
            logger.error(f"Error generating script: {str(e)}")
            raise

    def _validate_and_clean_script(self, script: List[Dict[str, str]]) -> List[Dict[str, str]]:
        cleaned = []
        expected_speaker = "Speaker 1"

        for item in script:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            speaker, dialogue = next(iter(item.items()))
            speaker = speaker.strip()
            if speaker not in ["Speaker 1", "Speaker 2"]:
                if "1" in speaker or "one" in speaker.lower():
                    speaker = "Speaker 1"
                elif "2" in speaker or "two" in speaker.lower():
                    speaker = "Speaker 2"
                else:
                    speaker = expected_speaker

            dialogue = dialogue.strip()
            if not dialogue:
                continue
            if not dialogue.endswith((".", "!", "?")):
                dialogue += "."

            cleaned.append({speaker: dialogue})
            expected_speaker = "Speaker 2" if expected_speaker == "Speaker 1" else "Speaker 1"

        if len(cleaned) < 2:
            raise ValueError("Generated script is too short or invalid")
        return cleaned



    def _extract_script_from_text(self, text: str) -> list:
        import re
        script = []
        lines = text.split("\n")
        current_speaker = None
        current_dialogue = []
        for line in lines:
            line = line.strip()
            m = re.match(r"Speaker\s*([12])\s*[:\-]\s*(.*)", line, re.IGNORECASE)
            if m:
                if current_speaker and current_dialogue:
                    dialogue = " ".join(current_dialogue).strip()
                    if dialogue and len(dialogue) >= 5:
                        if not dialogue.endswith((".", "!", "?")):
                            dialogue += "."
                        script.append({current_speaker: dialogue[:500]})
                current_speaker = "Speaker " + m.group(1)
                current_dialogue = [m.group(2).strip()] if m.group(2).strip() else []
            elif current_speaker and line:
                current_dialogue.append(line)
        if current_speaker and current_dialogue:
            dialogue = " ".join(current_dialogue).strip()
            if dialogue and len(dialogue) >= 5:
                if not dialogue.endswith((".", "!", "?")):
                    dialogue += "."
                script.append({current_speaker: dialogue[:500]})
        return script