import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Create a DataFrame with job descriptions and their categories
data = [
    ['Data Scientist with expertise in machine learning and Python.', 'Data Scientist'],
    ['Experienced project manager with skills in agile methodologies and team leadership.', 'Project Manager'],
    ['Full stack developer proficient in JavaScript, React, and Node.js.', 'Full Stack Developer'],
    ['Marketing expert with experience in digital marketing and SEO strategies.', 'Marketing Specialist']
]
df = pd.DataFrame(data, columns=['text', 'category'])

print("DataFrame with job descriptions and categories:")
print(df)

# Load the model
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

# Encode the job descriptions
text = df['text'].tolist()
vectors = encoder.encode(text)

print("\nEncoded vectors:")
print(vectors)

# Create FAISS index
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)

# Normalize vectors and add to index
faiss.normalize_L2(vectors)
index.add(vectors)
print("\nIndex info:", index.ntotal)

# Define a search query and encode it
search_text = 'Looking for a software engineer skilled in JavaScript and React.'
search_vector = encoder.encode(search_text)

# Convert the search_vector to a NumPy array and reshape for FAISS
_vector = np.array([search_vector]).astype(np.float32)

# Normalize the search vector
faiss.normalize_L2(_vector)

# Search in the index
k = index.ntotal  # Number of results to retrieve
distances, indices = index.search(_vector, k=k)

print("\nDistances:", distances)
print("Indices:", indices)

# Create a DataFrame to view results
results = pd.DataFrame({
    'Distance': distances[0],
    'Index': indices[0],
    'Job Description': df['text'].iloc[indices[0]].values,
    'Category': df['category'].iloc[indices[0]].values
})

print("\nSearch results DataFrame:")
print(results)

# Retrieve the category for the closest match
closest_index = indices[0][0]  # Get the index of the closest match
category = df['category'].iloc[closest_index]  # Retrieve category from the original DataFrame

print(f"\nThe category for the search text is: {category}")


