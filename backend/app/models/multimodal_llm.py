"""
Multimodal Large Language Model for Healthcare
Handles text, image, audio, and structured data
"""

import openai
import anthropic
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from PIL import Image
import whisper
import numpy as np
from typing import Dict, Any, List, Optional, Union
import base64
import io
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)

class MultimodalLLM:
    """
    Advanced multimodal LLM for healthcare applications
    Supports text, image, audio, and structured data processing
    """

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.whisper_model = None
        self.vision_model = None
        self.embedding_model = None
        self._initialize_models()
        logger.info("✅ Multimodal LLM initialized")

    def _initialize_models(self):
        """Initialize all multimodal models"""
        try:
            # OpenAI client
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ OpenAI client initialized")

            # Anthropic client
            if settings.ANTHROPIC_API_KEY:
                self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
                logger.info("✅ Anthropic client initialized")

            # Whisper for audio
            if settings.ENABLE_MULTIMODAL:
                try:
                    self.whisper_model = whisper.load_model("base")
                    logger.info("✅ Whisper model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Whisper model not loaded: {e}")

            # Vision model
            try:
                self.vision_model = pipeline("image-classification",
                                             model="microsoft/resnet-50")
                logger.info("✅ Vision model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Vision model not loaded: {e}")

            # Embedding model
            try:
                self.embedding_model = AutoModel.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("✅ Embedding model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Embedding model not loaded: {e}")

        except Exception as e:
            logger.error(f"❌ Multimodal LLM initialization error: {e}")

    async def process_multimodal_input(
            self,
            text: Optional[str] = None,
            image: Optional[str] = None,  # Base64 encoded
            audio: Optional[str] = None,  # Base64 encoded
            structured_data: Optional[Dict[str, Any]] = None,
            task: str = "general_analysis"
    ) -> Dict[str, Any]:
        """
        Process multimodal input and return comprehensive analysis
        """
        try:
            results = {
                "text_analysis": None,
                "image_analysis": None,
                "audio_analysis": None,
                "structured_analysis": None,
                "integrated_analysis": None,
                "task": task
            }

            # Process text
            if text:
                results["text_analysis"] = await self._process_text(text, task)

            # Process image
            if image and self.vision_model:
                results["image_analysis"] = await self._process_image(image, task)

            # Process audio
            if audio and self.whisper_model:
                results["audio_analysis"] = await self._process_audio(audio, task)

            # Process structured data
            if structured_data:
                results["structured_analysis"] = await self._process_structured_data(structured_data, task)

            # Integrate all modalities
            results["integrated_analysis"] = await self._integrate_modalities(results, task)

            return results

        except Exception as e:
            logger.error(f"Multimodal processing error: {e}")
            return {"error": str(e), "task": task}

    async def _process_text(self, text: str, task: str) -> Dict[str, Any]:
        """Process text input using LLM"""
        try:
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=settings.DEFAULT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a healthcare AI assistant. Task: {task}. Analyze the following text and provide medical insights."
                        },
                        {"role": "user", "content": text}
                    ],
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS
                )

                return {
                    "content": response.choices[0].message.content,
                    "model": settings.DEFAULT_MODEL,
                    "confidence": 0.9
                }

            return {"content": "Text analysis not available", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return {"error": str(e)}

    async def _process_image(self, image_b64: str, task: str) -> Dict[str, Any]:
        """Process image input"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))

            # Basic image analysis
            if self.vision_model:
                results = self.vision_model(image)

                # Enhanced analysis with OpenAI Vision
                if self.openai_client and "gpt-4" in settings.DEFAULT_MODEL:
                    response = await self.openai_client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Analyze this medical image for {task}. Provide detailed clinical observations."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_b64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=1000
                    )

                    return {
                        "vision_classification": results,
                        "detailed_analysis": response.choices[0].message.content,
                        "confidence": 0.85
                    }

                return {
                    "vision_classification": results,
                    "confidence": 0.7
                }

            return {"message": "Image analysis not available", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return {"error": str(e)}

    async def _process_audio(self, audio_b64: str, task: str) -> Dict[str, Any]:
        """Process audio input"""
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(audio_b64)

            # Save temporarily for Whisper
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name

            # Transcribe with Whisper
            if self.whisper_model:
                result = self.whisper_model.transcribe(temp_path)
                transcript = result["text"]

                # Analyze transcript for medical content
                if self.openai_client:
                    analysis = await self._process_text(transcript, f"audio_analysis_{task}")

                    return {
                        "transcript": transcript,
                        "language": result.get("language", "unknown"),
                        "analysis": analysis,
                        "confidence": 0.8
                    }

                return {
                    "transcript": transcript,
                    "language": result.get("language", "unknown"),
                    "confidence": 0.7
                }

            return {"message": "Audio analysis not available", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {"error": str(e)}
        finally:
            # Clean up temporary file
            try:
                import os
                os.unlink(temp_path)
            except:
                pass

    async def _process_structured_data(self, data: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Process structured healthcare data"""
        try:
            # Convert structured data to text for LLM analysis
            data_text = self._structured_to_text(data)

            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=settings.DEFAULT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Analyze this structured healthcare data for {task}. Provide clinical insights and recommendations."
                        },
                        {"role": "user", "content": data_text}
                    ],
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS
                )

                return {
                    "analysis": response.choices[0].message.content,
                    "data_summary": self._summarize_structured_data(data),
                    "confidence": 0.9
                }

            return {
                "data_summary": self._summarize_structured_data(data),
                "confidence": 0.6
            }

        except Exception as e:
            logger.error(f"Structured data processing error: {e}")
            return {"error": str(e)}

    async def _integrate_modalities(self, results: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Integrate insights from all modalities"""
        try:
            # Collect all available insights
            insights = []

            if results.get("text_analysis"):
                insights.append(f"Text Analysis: {results['text_analysis'].get('content', '')}")

            if results.get("image_analysis"):
                insights.append(f"Image Analysis: {results['image_analysis'].get('detailed_analysis', '')}")

            if results.get("audio_analysis"):
                insights.append(f"Audio Analysis: {results['audio_analysis'].get('transcript', '')}")

            if results.get("structured_analysis"):
                insights.append(f"Data Analysis: {results['structured_analysis'].get('analysis', '')}")

            if not insights:
                return {"message": "No modalities to integrate", "confidence": 0.0}

            # Use LLM to integrate insights
            if self.openai_client:
                integration_prompt = f"""
                Integrate the following multimodal healthcare analysis for {task}:
                
                {chr(10).join(insights)}
                
                Provide:
                1. Integrated clinical assessment
                2. Key findings across modalities
                3. Recommendations
                4. Confidence level
                """

                response = await self.openai_client.chat.completions.create(
                    model=settings.DEFAULT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert healthcare AI that integrates multimodal data for comprehensive patient assessment."
                        },
                        {"role": "user", "content": integration_prompt}
                    ],
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS
                )

                return {
                    "integrated_assessment": response.choices[0].message.content,
                    "modalities_used": len([r for r in results.values() if r and not isinstance(r, str)]),
                    "confidence": 0.9
                }

            return {
                "basic_integration": " | ".join(insights),
                "modalities_used": len(insights),
                "confidence": 0.6
            }

        except Exception as e:
            logger.error(f"Integration error: {e}")
            return {"error": str(e)}

    def _structured_to_text(self, data: Dict[str, Any]) -> str:
        """Convert structured data to readable text"""
        text_parts = []

        for key, value in data.items():
            if isinstance(value, (list, dict)):
                text_parts.append(f"{key}: {str(value)}")
            else:
                text_parts.append(f"{key}: {value}")

        return "\n".join(text_parts)

    def _summarize_structured_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize structured data"""
        return {
            "total_fields": len(data),
            "field_types": {k: type(v).__name__ for k, v in data.items()},
            "has_numeric_data": any(isinstance(v, (int, float)) for v in data.values()),
            "has_text_data": any(isinstance(v, str) for v in data.values())
        }

    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            if self.embedding_model and self.tokenizer:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                return embeddings

            # Fallback to OpenAI embeddings
            if self.openai_client:
                response = await self.openai_client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=text
                )
                return response.data[0].embedding

            return []

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return []
