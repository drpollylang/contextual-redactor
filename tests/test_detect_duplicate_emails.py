#!/usr/bin/env python3
"""
Test script for the find_stacked_email_duplicates method in utils.
This script analyzes a sample PDF document, extracts emails, and detects any duplicates within the emails - i.e. stacked
email threads with previous emails recursively nested within subsequent emails.
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from redactor.azure_client import AzureAIClient
from redactor.utils import extract_emails, find_stacked_email_duplicates, parse_thread, find_duplicate_emails

def test_find_email_duplicates(client, test_doc_filename: str):
    # Path to a test document containing NO emails
    # test_doc_path = os.path.join(os.path.dirname(__file__), '..', 'Test docs', 'CareActAssessmentTest.pdf')
    test_doc_path = os.path.join(os.path.dirname(__file__), '..', 'Test docs', test_doc_filename)

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

    # Extract emails
    try:
        # emails = extract_emails(result)
        emails = parse_thread(result.content)
    except Exception as e:
        print(f"Error extracting emails: {e}")
        return

    print(f"Found {len(emails)} emails.")

    # Find duplicate emails
    try:
        # duplicates = find_stacked_email_duplicates(emails, result)
        duplicates = find_duplicate_emails(emails)
    except Exception as e:
        print(f"Error detecting email duplicates: {e}")
        return

    # Print results
    for i, email in enumerate(emails):
        print(f"\nEmail {i+1}:")
        # print(f"  Content: {email['content'][:100]}...")  # Truncate for readability
        # # print(f"  Content: {email['content']}") 
        # print(f"  Span: {email['span']}")
        # print(f"  Bounding Region: {email['bounding_region']}")
        # --------
        # print(f"  From: {email['from_raw']}")
        # print(f"  To: {email['to_raw']} - {email['to']}")
        # print(f"  CC: {email['cc_raw']} - {email['cc']}")
        # print(f"  BCC: {email['bcc_raw']} - {email['bcc']}")
        # print(f"  subject: {email['subject']}")
        # print(f"  date: {email['date']}")
        # print(f"  message id: {email['message_id']}")
        # print(f"  in reply to: {email['in_reply_to']}")
        # print(f"  references: {email['references']}")
        # print(f"  content type: {email['content_type']}")
        # print(f"  body: {email['body']}")
        # print(f"  spans: {email['spans']}")
        # ------------------
        print(f"  Headers: {email['headers']}")
        print(f"  Body: {email['body']}")
        print(f"  Span: {email['span']}")
        print()

    for i, duplicate in enumerate(duplicates):
        print(f"\nDuplicate Email {i+1}:")
        print(f"  Source email index: {duplicate['source_email_index']}")
        print(f"  Target email index: {duplicate['target_email_index']}")
        print(f"  Type: {duplicate['type']}")
        try:
            print(f"  Coverage: {duplicate['coverage']}")
        except KeyError:
            pass
        print(f"  Content: {duplicate['content']}")
        print(f"  Target Span: {duplicate['target_span']}")
        print()
    
    duplicate_indices = set()
    for dup in duplicates:
        duplicate_indices.add(dup['target_email_index'])

    return duplicate_indices


def test_find_email_duplicates_wrapper():
    load_dotenv()
    
    # Initialize the client
    try:
        client = AzureAIClient()
    except Exception as e:
        print(f"Error initializing AzureAIClient: {e}")
        return

    # Test doc with no email duplicates
    print('Testing document with no email duplicates:')
    test_doc_no_dups = 'Telefonica Redaction Email Example 3 Disability.pdf'
    no_dups_test = test_find_email_duplicates(client, test_doc_no_dups)
    assert len(no_dups_test) == 0, f"Expected 0 emails, found {len(no_dups_test)}"

    # Test doc with stacked email thread - 3 duplicates expected
    print('-'*50 + '\nTesting document with stacked email thread containing 3 duplicates:')
    test_doc_with_dups = 'FakeStackedEmailThread.pdf'
    dups_test = test_find_email_duplicates(client, test_doc_with_dups)
    assert len(dups_test) == 3, f"Expected 3 duplicates, found {len(dups_test)}"


if __name__ == "__main__":
    test_find_email_duplicates_wrapper()