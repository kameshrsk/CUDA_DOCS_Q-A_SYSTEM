from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from retrieval import RetrievalSystem

from config import QA_MODEL_NAME, MAX_ANSWER_LENGTH

class QuestionAnswering:
    def __init__(self):
        self.retrieval_system = RetrievalSystem()
        print("Loading BART Model...")
        self.model = T5ForConditionalGeneration.from_pretrained(QA_MODEL_NAME)
        self.tokenizer = T5Tokenizer.from_pretrained(QA_MODEL_NAME)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device).half()
        self.model.eval() 
        print(f"Model Loaded Successfully on {self.device}")

    def answer_question(self, question, max_length=MAX_ANSWER_LENGTH):
        print(f"1. Preparing context for: {question}")
        context = " ".join(self.retrieval_system.retrieve_and_rerank(question, top_k=3))

        print("2. Tokenizing input...")
        inputs = self.tokenizer(
            f"question: {question} context: {context}",
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        print("3. Generating output...")
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"].to(self.device),
                num_beams=2, 
                min_length=20,
                max_length=max_length,
                early_stopping=True
            )

        print("4. Decoding output...")
        answer = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return answer

if __name__ == "__main__":
    qa_system = QuestionAnswering()

    questions = [
        "What is CUDA?",
        "What are the advantages of using CUDA for parallel computing?",
    ]

    for q in questions:
        answer = qa_system.answer_question(q)
        print(f"Question: {q}\nAnswer: {answer}\n")