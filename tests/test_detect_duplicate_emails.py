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
# from redactor.utils import extract_emails, find_stacked_email_duplicates, parse_thread, find_duplicate_emails
from redactor.redaction_logic import analyse_document_for_redactions, analyse_document_structure, analyse_document_for_duplicate_emails

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

    # # Extract emails
    # try:
    #     # emails = extract_emails(result)
    #     # emails = parse_thread(result.content)
    # except Exception as e:
    #     print(f"Error extracting emails: {e}")
    #     return

    # print(f"Found {len(emails)} emails.")

    # Find duplicate emails
    try:
        # duplicates = find_stacked_email_duplicates(emails, result)
        # duplicates = find_duplicate_emails(emails)
        duplicates = analyse_document_for_duplicate_emails(result)
    except Exception as e:
        print(f"Error detecting email duplicates: {e}")
        return

    # Print results
    # for i, email in enumerate(emails):
    #     print(f"\nEmail {i+1}:")
    #     # print(f"  Content: {email['content'][:100]}...")  # Truncate for readability
    #     # # print(f"  Content: {email['content']}") 
    #     # print(f"  Span: {email['span']}")
    #     # print(f"  Bounding Region: {email['bounding_region']}")
    #     # --------
    #     # print(f"  From: {email['from_raw']}")
    #     # print(f"  To: {email['to_raw']} - {email['to']}")
    #     # print(f"  CC: {email['cc_raw']} - {email['cc']}")
    #     # print(f"  BCC: {email['bcc_raw']} - {email['bcc']}")
    #     # print(f"  subject: {email['subject']}")
    #     # print(f"  date: {email['date']}")
    #     # print(f"  message id: {email['message_id']}")
    #     # print(f"  in reply to: {email['in_reply_to']}")
    #     # print(f"  references: {email['references']}")
    #     # print(f"  content type: {email['content_type']}")
    #     # print(f"  body: {email['body']}")
    #     # print(f"  spans: {email['spans']}")
    #     # ------------------
    #     print(f"  Headers: {email['headers']}")
    #     print(f"  Body: {email['body']}")
    #     print(f"  Span: {email['span']}")
    #     print()

    for i, duplicate in enumerate(duplicates):
        print(f"\nDuplicate Email {i+1}:")
        # print(f"  Source email index: {duplicate['source_email_index']}")
        # print(f"  Target email index: {duplicate['target_email_index']}")
        # print(f"  Type: {duplicate['type']}")
        # try:
        #     print(f"  Coverage: {duplicate['coverage']}")
        # except KeyError:
        #     pass
        # print(f"  Content: {duplicate['content']}")
        # print(f"  Target Span: {duplicate['target_span']}")
        print(f"  ID: {duplicate['id']}")
        print(f"  Text: {duplicate['text']}")
        print(f"  Category: {duplicate['category']}")
        print(f"  Reasoning: {duplicate['reasoning']}")
        print(f"  Context: {duplicate['context']}")
        print(f"  Rects: {duplicate['rects']}")
        print()
    
    duplicate_ids = set()
    for dup in duplicates:
        duplicate_ids.add(dup['id'].split('_')[-1]) # grab unique duplicates - remove copies due to multiple locations

    return duplicate_ids


def test_redaction_logic_funcs(test_doc_filename: str):
    test_doc_path = os.path.join(os.path.dirname(__file__), '..', 'Test docs', test_doc_filename)

    if not os.path.exists(test_doc_path):
        print(f"Test document not found: {test_doc_path}")
        return

    print(f"Analyzing document: {test_doc_path}")

    # Analyze the document
    try:
        result = analyse_document_structure(test_doc_path)
        # print(result.paragraphs[0].content)  # Debug print to verify content
    except Exception as e:
        print(f"Error analyzing document: {e}")
        return

    # Find duplicate emails
    try:
        duplicates = detect_email_duplicates(result)
    except Exception as e:
        print(f"Error detecting email duplicates: {e}")
        return

    # Print results
    for i, duplicate in enumerate(duplicates):
        print(f"Duplicate {i+1}:")
        # print(f"  Headers: {duplicate['headers']}")
        # print(f"  Body: {duplicate['body']}")
        # print(f"  Span: {duplicate['span']}")
        print(f"  ID: {duplicate['id']}")
        print(f"  Text: {duplicate['text']}")
        print(f"  Category: {duplicate['category']}")
        print(f"  Reasoning: {duplicate['reasoning']}")
        print(f"  Context: {duplicate['context']}")
        print(f"  Rects: {duplicate['rects']}\n")
    
    return duplicates


def test_find_email_duplicates_wrapper():
    load_dotenv()
    
    # Initialize the client
    try:
        client = AzureAIClient()
    except Exception as e:
        print(f"Error initializing AzureAIClient: {e}")
        return

    # Test doc with no email duplicates
    # print('Testing document with no email duplicates:')
    # test_doc_no_dups = 'Telefonica Redaction Email Example 3 Disability.pdf'
    # no_dups_test = test_find_email_duplicates(client, test_doc_no_dups)
    # assert len(no_dups_test) == 0, f"Expected 0 emails, found {len(no_dups_test)}"

    # Test doc with stacked email thread - 3 duplicates expected
    # print('-'*50 + '\nTesting document with stacked email thread containing 3 duplicates:')
    # test_doc_with_dups = 'FakeStackedEmailThread.pdf'
    # dups_test = test_find_email_duplicates(client, test_doc_with_dups)
    # assert len(dups_test) == 3, f"Expected 3 duplicates, found {len(dups_test)}"

    # Test functions in redaction_logic.py
    # print('-'*50 + '\nTesting document with stacked email thread containing 3 duplicates using redaction_logic.py funcs:')
    # test_doc_with_dups = 'FakeStackedEmailThread.pdf'
    # dups_redlog_test = test_redaction_logic_funcs(test_doc_with_dups)
    # assert len(dups_redlog_test) == 3, f"Expected 3 duplicates, found {len(dups_redlog_test)}"
    
    print('-'*50 + '\nTesting document with stacked email thread containing 3 duplicates:')
    test_doc_with_dups = 'FakeStackedEmailThread.pdf'
    dups_test = test_find_email_duplicates(client, test_doc_with_dups)
    assert len(dups_test) == 3, f"Expected 3 duplicates, found {len(dups_test)}"

    # Test docs with different styles/formatting of stacked email threads
    print('-'*50 + '\nTesting document with different stacked email thread format:')
    test_docs = [
        'email_duplicates_test_docs/testdoc_1_6.pdf', 
        'email_duplicates_test_docs/testdoc_2_6.pdf',
        'email_duplicates_test_docs/testdoc_3_6.pdf', 
        'email_duplicates_test_docs/testdoc_4_6.pdf'
        ]
    num_duplicates_expected = [6, 6, 6, 6]
    for i, test_doc in enumerate(test_docs):
        print(f'\nTesting document: {test_doc}')
        dups_test = test_find_email_duplicates(client, test_doc)
        assert len(dups_test) == num_duplicates_expected[i], f"Expected {num_duplicates_expected[i]} duplicates, found {len(dups_test)}"
        print("\n------------- ALL TESTS PASSED! -------------\n")

if __name__ == "__main__":
    test_find_email_duplicates_wrapper()