# CUDA Documentation Q&A System

This project implements a Question Answering (Q&A) system for CUDA documentation. It uses web crawling, document processing, vector database storage, and a neural language model to provide accurate answers to user queries about CUDA.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Components](#components)
6. [Configuration](#configuration)
7. [Requirements](#requirements)

## Overview

This Q&A system crawls the NVIDIA CUDA documentation, processes the content, stores it in a vector database, and uses a neural language model to generate answers to user queries. The system combines information retrieval and natural language processing techniques to provide accurate and context-aware responses.

## System Architecture

The system consists of the following main components:

1. Web Crawler: Scrapes CUDA documentation from the NVIDIA website.
2. Data Processor: Chunks the scraped text and creates embeddings.
3. Vector Database: Stores the text chunks and their embeddings for efficient retrieval.
4. Retrieval System: Fetches relevant document chunks based on the query.
5. Question Answering Model: Generates answers based on the retrieved context and user query.

## Installation

1. Clone this repository:
   git clone https://github.com/kameshrsk/CUDA_DOCS_Q-A_SYSTEM

2. Install the required packages:
   pip intstall -r requirements.txt

3. Install and start the Milvus vector database. Follow the official Milvus installation guide

## Usage

1. First, run the web crawler and create the vector database:

   python main.py

   Choose option 1 when prompted.

2. Once the database is created, you can start the Q&A system:

   Choose option 2 when prompted.

3. Enter your questions about CUDA when prompted. Type 'quit' to exit the system.

## Components

### Web Crawler (`web_crawler.py`)

Uses Scrapy to crawl the NVIDIA CUDA documentation website and extract relevant text content.

### Data Processor (`data_process.py`)

Processes the crawled data by chunking text, generating embeddings, and storing them in the Milvus vector database.

### Retrieval System (`retrieval.py`)

Implements a hybrid retrieval system using both vector similarity search and BM25 ranking. It also includes a re-ranking step for improved relevance.

### Question Answering (`question_answering.py`)

Uses a T5 model to generate answers based on the retrieved context and user query.

### Main Script (`main.py`)

Provides a command-line interface to run the web crawler, create the database, and interact with the Q&A system.

## Configuration

The `config.py` file contains various configuration parameters for the system, including:

- Milvus database settings
- Model names and parameters
- Web crawling settings
- Retrieval and re-ranking parameters

Modify this file to adjust the system's behavior without changing the core code.

## Requirements

The main requirements for this project are:

- Python 3.7+
- PyTorch
- Transformers
- Sentence-Transformers
- Scrapy
- PyMilvus
- NLTK
- scikit-learn
- rank_bm25
