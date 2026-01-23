import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
try:
    from annoy import AnnoyIndex
except ImportError:
    AnnoyIndex = None
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

from redactor.utils import log_ner_output

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentParagraph
from azure.ai.textanalytics import TextAnalyticsClient, PiiEntityCategory
from openai import AzureOpenAI, OpenAI

EMAIL_HEADER_PATTERN = re.compile(
    r"(from:.*?$|sent:.*?$|to:.*?$|subject:.*?$|cc:.*?$|bcc:.*?$)",
    flags=re.IGNORECASE | re.MULTILINE
)

def normalize_text(text: Optional[str]) -> str:
    """Normalize text for duplicate comparison (email-friendly)."""
    if not text:
        return ""
    t = EMAIL_HEADER_PATTERN.sub("", text)      # strip common headers
    t = re.sub(r"^>+\s?", "", t, flags=re.MULTILINE)  # remove quote markers
    t = " ".join(t.lower().split())         # lower + collapse whitespace
    return t

class AzureAIClient:
    def __init__(self):
        self.similarity_threshold = 0.99
        self.faiss_top_k = 5
        try:
            doc_intel_endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
            doc_intel_key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
            openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            openai_key = os.environ["AZURE_OPENAI_KEY"]
            self.openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
            self.embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
            self.openai_gpt4_mini_deployment = os.environ["AZURE_OPENAI_GPT41MINI_DEPLOYMENT_NAME"]
            self.openai_gpt41_mini_deplyment = os.environ["AZURE_OPENAI_GPT41MINI_DEPLOYMENT_NAME"]
            self.openai_gpt5_deployment = os.environ["AZURE_OPENAI_GPT5_DEPLOYMENT_NAME"]
            self.openai_gpt5_nano_deployment = os.environ["AZURE_OPENAI_GPT5NANO_DEPLOYMENT_NAME"]
            lang_endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
            lang_key = os.environ["AZURE_LANGUAGE_KEY"]

        except KeyError as e:
            raise RuntimeError(f"Environment variable not set: {e}") from e

        self.doc_intel_client = DocumentIntelligenceClient(
            endpoint=doc_intel_endpoint, credential=AzureKeyCredential(doc_intel_key)
        )
        self.openai_client = OpenAI(
            api_key=openai_key,
            base_url=openai_endpoint
            # api_version="2024-02-01",
            # azure_endpoint=openai_endpoint
        )
        self.text_analytics_client = TextAnalyticsClient(
            endpoint=lang_endpoint, credential=AzureKeyCredential(lang_key)
        )
        
    def analyse_document(self, file_path: str) -> AnalyzeResult:
        print("Analysing document with Azure AI Document Intelligence...")
        with open(file_path, "rb") as f:
            poller = self.doc_intel_client.begin_analyze_document(
                "prebuilt-layout", body=f, content_type="application/octet-stream"
            )
        result: AnalyzeResult = poller.result()
        print("Document analysis complete.")
        return result

    def parse_user_instructions(self, user_text: str) -> dict:
        """Uses an LLM to parse free-text instructions into a structured JSON object."""
        if not user_text or not user_text.strip():
            return {} # Return empty dict if there are no instructions

        system_prompt = """
        You are a configuration parser. Your task is to analyze the user's instructions for a document redaction tool and convert them into a structured JSON object.
        The JSON object should have two optional keys:
        1. "exceptions": A list of exact strings that the user wants to PREVENT from being redacted.
        2. "sensitive_content_rules": A single string describing any new, subjective content the user wants to find and redact.

        **CRITICAL RULE:** If you identify a multi-word person's name in the "exceptions" (e.g., "Oliver Hughes"), you MUST add BOTH the full name AND the first name to the exceptions list (e.g., ["Oliver Hughes", "Oliver"]). Do this only for names that look like people's names.

        If a category is not mentioned, omit its key from the JSON. Respond ONLY with the valid JSON object.

        --- EXAMPLES ---
        User Input: "keep sarah linton and oliver hughes, but also redact any mention of bullying"
        Your Output:
        {
        "exceptions": ["Sarah Linton", "Sarah", "Oliver Hughes", "Oliver"],
        "sensitive_content_rules": "Redact any mention of bullying."
        }
        ---
        User Input: "Don't remove the name Oliver Hughes"
        Your Output:
        {
        "exceptions": ["Oliver Hughes", "Oliver"]
        }
        ---
        User Input: "The company 'Hughes Construction' is fine to keep."
        Your Output:
        {
        "exceptions": ["Hughes Construction"]
        }
        ---
        User Input: "Find any quotes that are critical of the parents"
        Your Output:
        {
        "sensitive_content_rules": "Find any quotes that are critical of the parents"
        }
        """
        # Model 1 (parse user instructions)
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_gpt41_mini_deplyment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
                )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error parsing user instructions: {e}")
            return {} # Return empty on failure

    def get_pii(self, text_chunk: str, enable_log=True) -> list:
        """Extracts structured PII entities using Azure Language Studio."""
        comprehensive_pii_categories = [
                # Commented out categories are currently in preview and so cause errors when included.
                # Revisit this when upgrading to a newer version of the SDK (azure.ai.textanalytics >= 6.x).
                PiiEntityCategory.PERSON,
                PiiEntityCategory.PHONE_NUMBER,
                PiiEntityCategory.EMAIL,
                PiiEntityCategory.ADDRESS,
                PiiEntityCategory.DATE,
                PiiEntityCategory.AGE,
                PiiEntityCategory.IP_ADDRESS,
                PiiEntityCategory.UK_NATIONAL_INSURANCE_NUMBER,
                PiiEntityCategory.UK_NATIONAL_HEALTH_NUMBER,
                PiiEntityCategory.USUK_PASSPORT_NUMBER,
                PiiEntityCategory.UK_DRIVERS_LICENSE_NUMBER,
                # PiiEntityCategory.BANK_ACCOUNT_NUMBER,
                # PiiEntityCategory.SORT_CODE,
                PiiEntityCategory.CREDIT_CARD_NUMBER,
                PiiEntityCategory.UK_ELECTORAL_ROLL_NUMBER,
                PiiEntityCategory.UK_UNIQUE_TAXPAYER_NUMBER,
                PiiEntityCategory.ORGANIZATION,
                # PiiEntityCategory.LICENSE_PLATE
            ]

        # Create a confidence threshold override 
        # currently commented out because not supported by azure.ai.textanalytics version < 6.x (preview)
        # Example: Only return PERSON entities with confidence >= 0.4
        # threshold_override = ConfidenceScoreThresholdOverride(
        #     category=PiiEntityCategory.PERSON,
        #     threshold=0.4
        # )

        try:
            result = self.text_analytics_client.recognize_pii_entities(
                [text_chunk],
                categories_filter=comprehensive_pii_categories,
                # confidence_score_threshold_overrides=[threshold_override]
            )
            entities = [
                {"text": ent.text, 
                 "category": ent.category,
                 "confidence_score": ent.confidence_score,
                 "offset": ent.offset,
                 "length": ent.length                 
            }
                for doc in result if not doc.is_error for ent in doc.entities
            ]
            if enable_log:
                # print("Saving NER output to log...")
                for ent in entities:
                    log_ner_output(
                        "ner_log",
                        ent['text'], ent['category'], ent['confidence_score'], ent['offset'], ent['length'],
                        header=["text", "category", "confidence_score", "offset", "length"]
                    )
            return entities
        except Exception as e:
            print(f"Error getting PII from Language Service: {e}")
            return []
        
    def is_school(self, organization_name: str, context_sentence: str) -> bool:
        """
        Uses a cheap, fast LLM call to determine if an organization name is likely a school.
        """
        system_prompt = """
        You are a simple boolean classifier. Your only task is to determine if the given organization name is an educational institution (like a school, college, or university) based on the name and the context sentence it appeared in.
        Respond with a single word: "true" if it is an educational institution, and "false" if it is not.
        """
        user_prompt = f"Organization Name: \"{organization_name}\"\nContext Sentence: \"{context_sentence}\""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_gpt4_mini_deployment,
                #model=self.openai_gpt5_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1, # We only need one word
                temperature=0.0
            )
            # print(f"is_school check for '{organization_name}': {response.choices[0].message.content.strip()}")
            return response.choices[0].message.content.lower() == "true"
        except Exception as e:
            print(f"Error in is_school check: {e}")
            return False

    def link_entities_to_person(self, text_chunk: str, pii_entities: list) -> dict:
            """
            Uses an LLM to link PII entities to the primary person they belong to in the text.
            """
            if not pii_entities:
                return {}

            system_prompt = """
            You are an entity-linking specialist. Below is a block of text and a list of PII entities found within it.
            Your task is to link each PII entity to the primary person it belongs to in the text.
            Return a JSON object where the keys are the exact text of each PII entity, and the value is the name of the person it is associated with.
            - If an entity IS a person's name, the value should be the name itself.
            - If an entity clearly belongs to a person mentioned in the text, the value should be that person's name.
            - If an entity does not belong to any specific person, use the value 'None'.

            --- EXAMPLE ---
            Text: "Oliver (DOB: 14 March 2015) was quiet. He attends Bridgwater Primary School. Sarah Linton is the case worker."
            PII Entities: ["Oliver", "14 March 2015", "Bridgwater Primary School", "Sarah Linton"]

            Your Output:
            {
            "Oliver": "Oliver",
            "14 March 2015": "Oliver",
            "Bridgwater Primary School": "Oliver",
            "Sarah Linton": "Sarah Linton"
            }
            """
            # Format the PII entities for the user prompt
            entity_list_str = ", ".join([f'"{ent["text"]}"' for ent in pii_entities])
            user_prompt = f"Text: \"{text_chunk}\"\nPII Entities: [{entity_list_str}]"
            
            # Model 2 (entity linking)
            try:
                response = self.openai_client.chat.completions.create(
                    #model=self.openai_deployment,
                    model=self.openai_gpt4_mini_deployment,
                    #model=self.openai_gpt5_nano_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Error performing entity linking: {e}")
                return {}

    def get_sensitive_information(self, text_chunk: str, user_context: str) -> List[Dict]:
        """
        Uses an LLM for nuanced, context-aware redaction based on specific user rules.
        """

        system_prompt = f"""
        You are a highly advanced document analysis tool. Your task is to analyze a specific block of text based on a user's rule, using the surrounding text for context only.

        **USER'S SENSITIVE CONTENT RULE:** "{user_context}"

        --- YOUR THOUGHT PROCESS ---
        1. First, I will read the full text to understand the full context.
        2. Second, I will ONLY extract passages, sentences, or quotations from the "TARGET TEXT" that strictly match the user's rule. I will not extract anything from the context block.
        
        For each match, use the category `SensitiveContent`. In your reasoning, you MUST explain how the extracted text specifically relates to the user's rule.

        CRITICAL: Only extract text that directly matches the user's rule. Do not extract anything else.

        **Output Format:**
        Respond ONLY with a valid JSON object with a single key "redactions", which is an array of objects.
        Each object must have "text", "category", and "reasoning". If nothing is found, return an empty "redactions" array.
        """
        # Model 3 (sensitive content analysis)
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                #model=self.openai_gpt4_mini_deployment,
                #model=self.openai_gpt5_nano_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_chunk}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            response_content = response.choices[0].message.content
            if response_content:
                data = json.loads(response_content)
                return data.get("redactions", [])
            return []
        except Exception as e:
            print(f"An error occurred while calling Azure OpenAI: {e}")
            return []


    def _embed_paragraphs_batch(self, paragraphs: List[DocumentParagraph]) -> np.ndarray:
        """Embed a batch of DocumentParagraph objects using Azure OpenAI - for duplicate detection using cosine similarity."""
        inputs = [normalize_text(getattr(p, "content", None) or "") for p in paragraphs]
        if not any(inputs):
            return np.zeros((len(paragraphs), 1536), dtype=np.float32)
        try:
            resp = self.openai_client.embeddings.create(
                model=self.embedding_deployment,
                input=inputs
            )
            vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
            return np.vstack(vecs).astype(np.float32) if vecs else np.zeros((len(paragraphs), 1536), dtype=np.float32)
        except Exception as e:
            print(f"Error embedding paragraphs: {e}")
            return np.zeros((len(paragraphs), 1536), dtype=np.float32)


    def detect_duplicates(self, text_blocks: list[DocumentParagraph], method: str = 'annoy') -> List[List[Dict[str, Any]]]:
        """
        Run exact and near-duplicate matching for each DocumentParagraph in the analysed document.
        Uses either scikit-learn (exact cosine similarity) or Annoy (approximate) for similarity search.
        
        Args:
            text_blocks: List of DocumentParagraph objects.
            method: 'sklearn' for exact similarity, 'annoy' for approximate using Annoy.
        
        Returns:
            A list of lists of dicts, where each item is a matched set of duplicates, each of
            which contains a list of dicts, each dict representing information about the DocumentParagraph
            that is potentially duplicated.
        """
        if not text_blocks:
            return []

        # Embed all paragraphs in batches
        batch_size = 64
        all_vecs = []
        for i in range(0, len(text_blocks), batch_size):
            batch = text_blocks[i:i + batch_size]
            vecs = self._embed_paragraphs_batch(batch)
            all_vecs.append(vecs)
        vectors = np.vstack(all_vecs).astype(np.float32) if all_vecs else np.zeros((0, 1536), dtype=np.float32)

        if vectors.shape[0] == 0:
            return []

        # Choose similarity method
        if method == 'sklearn':
            sim_matrix = cosine_similarity(vectors)
            pairs = []
            n = vectors.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(sim_matrix[i, j])
                    if sim >= self.similarity_threshold:
                        pairs.append((i, j, sim))
        elif method == 'annoy':
            if AnnoyIndex is None:
                raise ImportError("Annoy is not installed. Install with 'pip install annoy' or 'poetry add annoy'")
            f = vectors.shape[1]
            index = AnnoyIndex(f, 'angular')
            for i, vec in enumerate(vectors):
                index.add_item(i, vec)
            index.build(10)  # n_trees
            pairs = []
            n = vectors.shape[0]
            for i in range(n):
                neighbors, distances = index.get_nns_by_item(i, self.faiss_top_k + 1, include_distances=True)
                for j, dist in zip(neighbors, distances):
                    if j > i:
                        sim = 1 - (dist ** 2) / 2  # convert angular distance to cosine similarity
                        if sim >= self.similarity_threshold:
                            pairs.append((i, j, sim))
        else:
            raise ValueError("Invalid method. Choose 'sklearn' or 'annoy'")

        # Group into duplicate sets using union-find
        parent = list(range(len(text_blocks)))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j, _ in pairs:
            union(i, j)

        # Collect groups
        groups = {}
        for i in range(len(text_blocks)):
            p = find(i)
            if p not in groups:
                groups[p] = []
            groups[p].append(i)

        # Build output: list of lists of dicts
        duplicate_groups = []
        for group_indices in groups.values():
            if len(group_indices) > 1:  # only groups with more than one
                group_dicts = []
                for idx in group_indices:
                    para = text_blocks[idx]
                    group_dicts.append({
                        "index": idx,
                        "content": getattr(para, "content", None) or "",
                        "bounding_regions": getattr(para, "bounding_regions", None) or []
                    })
                duplicate_groups.append(group_dicts)

        return duplicate_groups