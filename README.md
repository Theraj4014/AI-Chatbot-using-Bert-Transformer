# DoctorGPT: Medical Advice Chatbot Using BERT

DoctorGPT is a chatbot that uses BERT to provide health-related advice based on user symptoms. The model retrieves advice for various diseases using disease embeddings and a pre-trained BERT model.

## Features
- **Symptom-based Disease Detection**: Utilizes BERT for classifying diseases based on input text.
- **Medical Advice Retrieval**: Fetches health advice for various diseases using similarity search based on precomputed embeddings.
- **Interactive Web Application**: Built using Dash for smooth user interaction.

## Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required Libraries:
  ```bash
  pip install torch transformers scikit-learn dash pandas numpy


## Project Structure
- **app.py**: Main Python script that initializes and runs the Dash web application.
saved_bert_model_and_tokenizer/: Directory containing the pre-trained BERT model, tokenizer, and other essential files.
- **corpus.txt**: A text file with a corpus of health-related advice.
- **corpus_embeddings.npy**: Precomputed embeddings of the advice corpus.
- **f_label.pkl**: Pickle file containing the disease labels.



## Key Concepts and Functionality
### BERT-Based Disease Classification
DoctorGPT utilizes a fine-tuned BERT model that helps in identifying possible diseases from user input. It analyzes the input symptoms, processes the text, and suggests relevant diseases.

### Medical Advice Retrieval Using Embeddings
The advice retrieval system is based on cosine similarity, where precomputed advice embeddings are compared with user input embeddings to find the closest match. This way, DoctorGPT offers relevant health advice for the predicted disease.

### Visualization of Advice Data
DoctorGPT includes a feature that visualizes the advice corpus using a word cloud. This helps in representing key terms associated with different diseases and their related advice.

### Example Input and Output
Input: "I have a sore throat and fatigue."
Output: DoctorGPT might predict diseases like "Flu" or "Common Cold" and provide advice, such as:
Drink plenty of fluids
Get enough rest
Consider seeing a doctor if symptoms persist
Model Saving Path
The BERT model and embeddings are saved at the path /content/gdrive/MyDrive/medical_chatbot/data/nlp_chatbot. You can save or update the model at this location during training or inference.

### Visualization of Key Terms
DoctorGPT also supports creating a word cloud based on the most frequent terms found in the health advice data. This visualization helps users understand the focus areas of the provided advice.
