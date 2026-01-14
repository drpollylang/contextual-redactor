#!/usr/bin/env python3
"""
Test script for the detect_duplicates method in AzureAIClient.
This script analyzes a sample PDF document, extracts paragraphs, and detects duplicates.
If Azure credentials are not set, it runs a mock test with predefined paragraphs.
"""

import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from redactor.azure_client import AzureAIClient
from redactor.redaction_logic import merge_small_paragraphs
from azure.ai.documentintelligence.models import DocumentParagraph

def create_mock_paragraphs():
    """Create mock DocumentParagraph objects for testing."""
    # Create some duplicate and non-duplicate paragraphs
    paragraphs = [
        DocumentParagraph(content="This is a test paragraph."),
        DocumentParagraph(content="This is a test paragraph."),  # duplicate
        DocumentParagraph(content="Another paragraph with different content."),
        DocumentParagraph(content="This is a test paragraph."),  # another duplicate
        DocumentParagraph(content="Yet another unique paragraph."),
    ]
    return paragraphs

def test_detect_duplicates_mock(client):
    """Test with mock data."""
    print("Running mock test (no Azure credentials required).")
    paragraphs = create_mock_paragraphs()
    print(f"Created {len(paragraphs)} mock paragraphs.")

    # Since embeddings are needed, but for mock, we can skip or use dummy
    # For now, just call the method and see if it handles empty vectors
    try:
        duplicates = client.detect_duplicates(paragraphs)
        print(f"Mock test completed. Found {len(duplicates)} duplicate groups.")
        # Print results
        for i, group in enumerate(duplicates):
            print(f"\nDuplicate Group {i+1}:")
            for para in group:
                print(f"  Index: {para['index']}")
                # print(f"  Content: {para['content'][:100]}...")  # Truncate for readability
                print(f"  Content: {para['content']}") 
                print(f"  Bounding Regions: {para['bounding_regions']}")
                print()
        # Note: With real embeddings, this would find duplicates
    except Exception as e:
        print(f"Error in mock test: {e}")

def test_detect_duplicates_real(client):
    """Test with real document analysis."""
    # Path to a test document
    test_doc_path = os.path.join(os.path.dirname(__file__), '..', 'Test docs', 'CareActAssessmentTest.pdf')

    if not os.path.exists(test_doc_path):
        print(f"Test document not found: {test_doc_path}")
        return

    print(f"Analyzing document: {test_doc_path}")

    # Analyze the document
    try:
        result = client.analyse_document(test_doc_path)
    except Exception as e:
        print(f"Error analyzing document: {e}")
        return

    # Extract and merge small paragraphs
    paragraphs = merge_small_paragraphs(result.paragraphs)
    if not paragraphs:
        print("No paragraphs found in the document.")
        return

    print(f"Found {len(paragraphs)} paragraphs.")

    # Detect duplicates
    try:
        duplicates = client.detect_duplicates(paragraphs)
    except Exception as e:
        print(f"Error detecting duplicates: {e}")
        return

    print(f"Found {len(duplicates)} duplicate groups.")

    # Print results
    for i, group in enumerate(duplicates):
        print(f"\nDuplicate Group {i+1}:")
        for para in group:
            print(f"  Index: {para['index']}")
            # print(f"  Content: {para['content'][:100]}...")  # Truncate for readability
            print(f"  Content: {para['content']}") 
            print(f"  Bounding Regions: {para['bounding_regions']}")
            print()

def test_detect_duplicates():
    load_dotenv()
    # Check if Azure credentials are set
    # required_env_vars = [
    #     "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    #     "AZURE_DOCUMENT_INTELLIGENCE_KEY",
    #     "AZURE_OPENAI_ENDPOINT",
    #     "AZURE_OPENAI_KEY",
    #     "AZURE_OPENAI_DEPLOYMENT_NAME",
    #     "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
    #     "AZURE_OPENAI_GPT35_DEPLOYMENT_NAME",
    #     "AZURE_OPENAI_GPT41MINI_DEPLOYMENT_NAME",
    #     "AZURE_OPENAI_GPT5_DEPLOYMENT_NAME",
    #     "AZURE_OPENAI_GPT5NANO_DEPLOYMENT_NAME",
    #     "AZURE_LANGUAGE_ENDPOINT",
    #     "AZURE_LANGUAGE_KEY"
    # ]

    # missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    # if missing_vars:
    #     print(f"Missing environment variables: {', '.join(missing_vars)}")
    #     print("Running mock test instead.")
    #     client = AzureAIClient()  # This will fail, but let's catch
    #     try:
    #         client = AzureAIClient()
    #         test_detect_duplicates_mock(client)
    #     except Exception as e:
    #         print(f"Cannot initialize client: {e}")
    #     return

    # Initialize the client
    try:
        client = AzureAIClient()
    except Exception as e:
        print(f"Error initializing AzureAIClient: {e}")
        return

    # Mock test
    test_detect_duplicates_mock(client)

    # Real document test
    test_detect_duplicates_real(client)

if __name__ == "__main__":
    test_detect_duplicates()