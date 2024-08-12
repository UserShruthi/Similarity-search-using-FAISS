# Similarity-search-using-FAISS
This repository contains a prototype for performing similarity search using FAISS (Facebook AI Similarity Search) and Sentence Transformers. The project demonstrates how to build an efficient similarity search system for job descriptions based on vector embeddings.

## Overview

FAISS is a library developed by Facebook AI for efficient similarity search and clustering of dense vectors. In this project, FAISS is used to perform similarity searches on job descriptions encoded into vectors using Sentence Transformers. The goal is to match a search query with relevant job descriptions and identify their categories based on similarity.

## Features

- **Vector Embeddings**: Convert job descriptions into numerical vectors using Sentence Transformers.
- **Similarity Search**: Use FAISS to search for the most similar job descriptions based on vector similarity.
- **Normalization**: Ensure consistent scaling of vectors for accurate similarity measurement.
- **Search Results**: Retrieve and display the most similar job descriptions and their categories.

## Getting Started

### Prerequisites

You will need Python and the following libraries installed:

- `pandas`
- `numpy`
- `sentence-transformers`
- `faiss`
