from web_crawler import run_spider
from data_process import process_and_store
from question_answering import QuestionAnswering
from pymilvus import connections, utility

from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

def check_database_exists():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        return utility.has_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        return False

def crawl_and_create_db():
    print("Starting web crawling...")
    run_spider()
    print("Web crawling completed")

    print("Processing and creating Vector DB...")
    process_and_store()
    print("Data processing and storing completed.")

def run_qa_system():
    if not check_database_exists():
        print("Database not found. Please create the database first.")
        return

    print("Initializing Question Answer system...")
    qa_system = QuestionAnswering()

    while True:
        question = input("Enter the question (or enter 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        answer = qa_system.answer_question(question)
        print(f"\nQuestion: {question}\nAnswer: {answer}\n")

    print("Thank you for using the Q&A system")

def main():
    while True:
        print("\nCUDA Documentation Q&A System")
        print("1. Crawl and create database")
        print("2. Run Q&A system")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            crawl_and_create_db()
        elif choice == '2':
            run_qa_system()
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()