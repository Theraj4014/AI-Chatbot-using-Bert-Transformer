import torch
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load medical model, tokenizer, and labels
input_dir = 'saved_bert_model_and_tokenizer/'
loaded_model = BertForSequenceClassification.from_pretrained(input_dir)
loaded_model.eval()
loaded_tokenizer = BertTokenizer.from_pretrained(input_dir)
loaded_df_label = pd.read_pickle('f_label.pkl')

# Load advice retrieval model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load corpus and precomputed embeddings
with open('corpus.txt', 'r') as file:
    corpus = [line.strip() for line in file.readlines()]

corpus_embeddings = np.load('corpus_embeddings.npy')
disease_dict = {
    "Influenza": {
        "Symptoms": ["Fever", "Cough", "Sore throat", "Body aches", "Fatigue", "Chills"],
        "Advice": ["Rest and hydrate", "Over-the-counter pain relievers", "Antiviral medications (if prescribed)", "Stay isolated to prevent spreading"]
    },
    "COVID-19": {
        "Symptoms": ["Fever or chills", "Cough", "Shortness of breath", "Fatigue", "Loss of taste or smell", "Muscle aches"],
        "Advice": ["Isolate yourself", "Seek medical attention if symptoms worsen", "Stay hydrated", "Follow local health guidelines"]
    },
    "Malaria": {
        "Symptoms": ["Fever", "Chills", "Sweating", "Headache", "Nausea", "Muscle pain"],
        "Advice": ["Antimalarial medications (prescribed)", "Rest and hydrate", "Prevent mosquito bites with nets and repellents"]
    },
    "Tuberculosis": {
        "Symptoms": ["Persistent cough", "Coughing up blood", "Chest pain", "Fatigue", "Weight loss", "Night sweats"],
        "Advice": ["Complete antibiotic course as prescribed", "Regular follow-ups with a healthcare provider", "Avoid close contact with others"]
    },
    "Hepatitis B": {
        "Symptoms": ["Fatigue", "Abdominal pain", "Dark urine", "Jaundice", "Loss of appetite"],
        "Advice": ["Antiviral medications (if chronic)", "Avoid alcohol", "Regular monitoring by a healthcare provider"]
    },
    "Dengue": {
        "Symptoms": ["High fever", "Severe headache", "Pain behind the eyes", "Joint and muscle pain", "Nausea", "Rash"],
        "Advice": ["Hydrate with fluids", "Pain relievers (avoid NSAIDs)", "Seek medical attention if severe symptoms develop"]
    },
    "Cholera": {
        "Symptoms": ["Severe watery diarrhea", "Vomiting", "Leg cramps", "Dehydration", "Dry mucous membranes"],
        "Advice": ["Oral rehydration solutions", "Antibiotics (if severe)", "Seek immediate medical help"]
    },
    "Rabies": {
        "Symptoms": ["Fever", "Headache", "Nausea", "Agitation", "Anxiety", "Confusion", "Hydrophobia"],
        "Advice": ["Immediate post-exposure prophylaxis (PEP)", "Vaccination before symptoms appear"]
    },
    "Yellow Fever": {
        "Symptoms": ["Fever", "Chills", "Headache", "Back pain", "Fatigue", "Jaundice"],
        "Advice": ["Vaccination for prevention", "Supportive care (hydration, rest)"]
    },
    "Ebola": {
        "Symptoms": ["Fever", "Severe headache", "Muscle pain", "Vomiting", "Diarrhea", "Unexplained bleeding"],
        "Advice": ["Supportive care in a healthcare facility", "Experimental treatments (if available)"]
    },
    "Zika Virus": {
        "Symptoms": ["Fever", "Rash", "Joint pain", "Red eyes", "Muscle pain", "Headache"],
        "Advice": ["Hydration", "Rest", "Pain relievers (avoid NSAIDs)"]
    },
    "HIV/AIDS": {
        "Symptoms": ["Fever", "Chills", "Night sweats", "Muscle aches", "Fatigue", "Swollen lymph nodes"],
        "Advice": ["Antiretroviral therapy (ART)", "Regular medical care", "Healthy lifestyle"]
    },
    "Syphilis": {
        "Symptoms": ["Sores", "Rash", "Fever", "Swollen lymph nodes", "Fatigue", "Headache"],
        "Advice": ["Antibiotics (usually penicillin)", "Regular follow-ups"]
    },
    "Gonorrhea": {
        "Symptoms": ["Painful urination", "Abnormal discharge", "Lower abdominal pain", "Testicular swelling"],
        "Advice": ["Antibiotic treatment", "Inform sexual partners", "Regular STI screenings"]
    },
    "Chlamydia": {
        "Symptoms": ["Painful urination", "Abnormal discharge", "Pelvic pain", "Pain during intercourse"],
        "Advice": ["Antibiotic treatment", "Regular screenings for sexually active individuals"]
    },
    "Herpes": {
        "Symptoms": ["Blisters or sores", "Itching", "Burning sensation", "Pain during urination", "Fever"],
        "Advice": ["Antiviral medications", "Avoiding triggers and practicing safe sex"]
    },
    "HPV": {
        "Symptoms": ["Warts on genitals", "Itching", "Discomfort", "Warts in the throat"],
        "Advice": ["Vaccination for prevention", "Regular screenings for cervical cancer"]
    },
    "Mumps": {
        "Symptoms": ["Swollen salivary glands", "Fever", "Headache", "Muscle aches", "Fatigue"],
        "Advice": ["Rest, hydration, and over-the-counter pain relievers", "Vaccination for prevention"]
    },
    "Measles": {
        "Symptoms": ["Fever", "Dry cough", "Runny nose", "Sore throat", "Conjunctivitis", "Rash"],
        "Advice": ["Supportive care (hydration, fever management)", "Vaccination for prevention"]
    },
    "Rubella": {
        "Symptoms": ["Mild fever", "Headache", "Rash", "Swollen lymph nodes", "Muscle pain"],
        "Advice": ["Supportive care, vaccination for prevention"]
    },
    "Chickenpox": {
        "Symptoms": ["Fever", "Itchy rash", "Blisters", "Fatigue", "Headache"],
        "Advice": ["Antihistamines for itching, antiviral medications (if severe)", "Vaccination for prevention"]
    },
    "Shingles": {
        "Symptoms": ["Pain", "Burning sensation", "Itching", "Rash", "Blisters"],
        "Advice": ["Antiviral medications", "Pain management", "Vaccination for prevention"]
    },
    "Whooping Cough": {
        "Symptoms": ["Severe cough", "Whooping sound", "Vomiting after coughing", "Fatigue", "Fever"],
        "Advice": ["Antibiotics, supportive care, vaccination for prevention"]
    },
    "Diphtheria": {
        "Symptoms": ["Sore throat", "Fever", "Weakness", "Swollen glands", "Thick coating in throat"],
        "Advice": ["Antibiotics, antitoxin administration, vaccination for prevention"]
    },
    "Tetanus": {
        "Symptoms": ["Muscle stiffness", "Spasms", "Difficulty swallowing", "Jaw stiffness", "Fever"],
        "Advice": ["Tetanus vaccine (if not up to date), supportive care"]
    },
    "Polio": {
        "Symptoms": ["Fever", "Fatigue", "Headache", "Vomiting", "Stiff neck", "Paralysis"],
        "Advice": ["Supportive care, vaccination for prevention"]
    },
    "Meningitis": {
        "Symptoms": ["Fever", "Headache", "Stiff neck", "Nausea", "Vomiting", "Sensitivity to light"],
        "Advice": ["Immediate medical care, antibiotics or antiviral medications as needed"]
    },
    "Pneumonia": {
        "Symptoms": ["Fever", "Cough", "Shortness of breath", "Chest pain", "Fatigue"],
        "Advice": ["Antibiotics (if bacterial), rest, hydration"]
    },
    "Bronchitis": {
        "Symptoms": ["Cough", "Mucus production", "Fatigue", "Shortness of breath", "Chest discomfort"],
        "Advice": ["Rest, hydration, and over-the-counter medications for symptom relief"]
    },
    "Asthma": {
        "Symptoms": ["Shortness of breath", "Chest tightness", "Wheezing", "Coughing"],
        "Advice": ["Inhalers (bronchodilators), avoiding triggers, maintaining a healthy lifestyle"]
    },
    "COPD": {
        "Symptoms": ["Shortness of breath", "Wheezing", "Chest tightness", "Chronic cough"],
        "Advice": ["Bronchodilators, pulmonary rehabilitation, lifestyle changes (quitting smoking)"]
    },
    "Lung Cancer": {
        "Symptoms": ["Coughing", "Chest pain", "Shortness of breath", "Weight loss", "Fatigue"],
        "Advice": ["Medical treatment (surgery, chemotherapy, radiation), supportive care"]
    },
    "Breast Cancer": {
        "Symptoms": ["Lump in breast", "Change in breast shape", "Nipple discharge", "Skin dimpling"],
        "Advice": ["Surgery, chemotherapy, radiation, hormone therapy depending on stage"]
    },
    "Prostate Cancer": {
        "Symptoms": ["Difficulty urinating", "Blood in urine", "Pelvic discomfort", "Bone pain"],
        "Advice": ["Medical treatment (surgery, radiation, hormone therapy), regular monitoring"]
    },
    "Skin Cancer": {
        "Symptoms": ["New mole", "Change in existing mole", "Itching", "Bleeding"],
        "Advice": ["Surgical removal, radiation therapy, chemotherapy depending on type and stage"]
    },
    "Diabetes": {
        "Symptoms": ["Frequent urination", "Increased thirst", "Extreme fatigue", "Blurred vision"],
        "Advice": ["Blood sugar monitoring, insulin or medication as prescribed, dietary management"]
    },
    "Hypertension": {
        "Symptoms": ["Headache", "Shortness of breath", "Nosebleeds", "Flushing"],
        "Advice": ["Blood pressure monitoring, lifestyle changes (diet, exercise), medications as prescribed"]
    },
    "Heart Disease": {
        "Symptoms": ["Chest pain", "Shortness of breath", "Fatigue", "Palpitations"],
        "Advice": ["Medical treatment (medications, lifestyle changes, surgery if needed)"]
    },
    "Stroke": {
        "Symptoms": ["Sudden numbness", "Confusion", "Difficulty speaking", "Loss of balance"],
        "Advice": ["Seek immediate medical attention (call emergency services), rehabilitation therapy as needed"]
    },
    "Kidney Disease": {
        "Symptoms": ["Fatigue", "Swelling", "Changes in urination", "High blood pressure"],
        "Advice": ["Regular medical check-ups, dietary management, medications as prescribed"]
    },
    "Liver Disease": {
        "Symptoms": ["Fatigue", "Jaundice", "Swelling", "Loss of appetite"],
        "Advice": ["Avoid alcohol, regular monitoring by a healthcare provider, medications as prescribed"]
    },
    "Gallbladder Disease": {
        "Symptoms": ["Abdominal pain", "Nausea", "Vomiting", "Bloating"],
        "Advice": ["Dietary changes, possible surgery, medications as prescribed"]
    },
    "Thyroid Disorders": {
        "Symptoms": ["Fatigue", "Weight changes", "Temperature sensitivity", "Mood changes"],
        "Advice": ["Thyroid hormone replacement (for hypothyroidism), regular monitoring, medication as prescribed"]
    },
    "Anemia": {
        "Symptoms": ["Fatigue", "Pale skin", "Shortness of breath", "Dizziness"],
        "Advice": ["Iron supplements, dietary changes, treat underlying causes"]
    },
    "Osteoporosis": {
        "Symptoms": ["Bone fractures", "Loss of height", "Back pain", "Posture changes"],
        "Advice": ["Calcium and vitamin D supplements, weight-bearing exercises, medications as prescribed"]
    },
    "Arthritis": {
        "Symptoms": ["Joint pain", "Stiffness", "Swelling", "Fatigue"],
        "Advice": ["Pain relievers, physical therapy, lifestyle changes"]
    },
    "Allergies": {
        "Symptoms": ["Sneezing", "Itching", "Runny nose", "Rashes", "Swelling"],
        "Advice": ["Avoid allergens, antihistamines, consult an allergist"]
    },
    "Autoimmune Diseases": {
        "Symptoms": ["Fatigue", "Joint pain", "Skin rashes", "Fever"],
        "Advice": ["Immunosuppressive medications, regular monitoring, lifestyle management"]
    },
    "Mental Health Disorders": {
        "Symptoms": ["Mood changes", "Anxiety", "Fatigue", "Sleep disturbances"],
        "Advice": ["Therapy, medications as prescribed, support groups"]
    },
    "Gastroesophageal Reflux Disease (GERD)": {
        "Symptoms": ["Heartburn", "Regurgitation", "Difficulty swallowing", "Chest pain"],
        "Advice": ["Dietary changes, antacids, medications as prescribed"]
    },
    "Irritable Bowel Syndrome (IBS)": {
        "Symptoms": ["Abdominal pain", "Bloating", "Changes in bowel habits"],
        "Advice": ["Dietary management, stress reduction, medications as prescribed"]
    },
    "Celiac Disease": {
        "Symptoms": ["Diarrhea", "Weight loss", "Bloating", "Fatigue"],
        "Advice": ["Strict gluten-free diet, regular monitoring"]
    },
    "Crohn's Disease": {
        "Symptoms": ["Abdominal pain", "Diarrhea", "Fatigue", "Weight loss"],
        "Advice": ["Medications (anti-inflammatories), dietary changes, possible surgery"]
    },
    "Ulcerative Colitis": {
        "Symptoms": ["Abdominal pain", "Diarrhea (often bloody)", "Fatigue"],
        "Advice": ["Medications (anti-inflammatories), dietary management, possible surgery"]
    },
    "Diverticulitis": {
        "Symptoms": ["Abdominal pain", "Fever", "Nausea", "Change in bowel habits"],
        "Advice": ["Antibiotics, dietary changes, possible surgery if severe"]
    },
    "Hernia": {
        "Symptoms": ["Bulging in the abdomen", "Pain", "Discomfort"],
        "Advice": ["Surgery may be required, avoid heavy lifting"]
    },
    "Kidney Stones": {
        "Symptoms": ["Severe pain", "Blood in urine", "Nausea", "Frequent urination"],
        "Advice": ["Increased hydration, pain management, possible medical procedures"]
    },
    "Gout": {
        "Symptoms": ["Severe pain in joints", "Swelling", "Redness", "Limited mobility"],
        "Advice": ["Medications (anti-inflammatories, uric acid-lowering), dietary changes"]
    },
    "Psoriasis": {
        "Symptoms": ["Red patches", "Itching", "Thickened skin", "Dry, cracked skin"],
        "Advice": ["Topical treatments, phototherapy, systemic medications"]
    },
    "Eczema": {
        "Symptoms": ["Dry skin", "Itching", "Red, inflamed patches", "Blisters"],
        "Advice": ["Moisturizers, topical steroids, avoiding triggers"]
    },
    "Dermatitis": {
        "Symptoms": ["Redness", "Itching", "Swelling", "Dry patches"],
        "Advice": ["Avoid irritants, topical treatments, antihistamines for itching"]
    },
    "Skin Infections": {
        "Symptoms": ["Redness", "Swelling", "Pain", "Pus formation"],
        "Advice": ["Antibiotics (if bacterial), keep the area clean and dry"]
    },
    "Fungal Infections": {
        "Symptoms": ["Red, itchy patches", "Flaky skin", "Discomfort"],
        "Advice": ["Antifungal treatments, keep the area dry"]
    },
    "Bacterial Infections": {
        "Symptoms": ["Fever", "Redness, swelling, pain", "Pus formation"],
        "Advice": ["Antibiotics as prescribed, proper hygiene"]
    },
    "Viral Infections": {
        "Symptoms": ["Fever", "Fatigue", "Muscle aches", "Rashes"],
        "Advice": ["Rest, hydration, supportive care"]
    }
}

# Function to detect medical symptoms
def medical_symptom_detector(intent):
    pt_batch = loaded_tokenizer(
        intent,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    pt_outputs = loaded_model(**pt_batch)
    _, id = torch.max(pt_outputs[0], dim=1)
    prediction = loaded_df_label.iloc[[id.item()]]['intent'].item()
    return prediction

# Function to create embeddings for a query
def create_embedding(query):
    inputs = bert_tokenizer(query, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.numpy()

# Function to retrieve top matches from advice corpus
def retrieve_top_matches(query_embedding, corpus_embeddings, top_k=10):
    similarities = cosine_similarity(query_embedding, corpus_embeddings)
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    return top_indices, similarities[0][top_indices]

# Create Dash app
app = Dash(__name__)

# Styles for chatbot interface
styles = {
    'page': {
        'display': 'flex', 'height': '100vh', 'backgroundColor': '#f0f0f5',
        'fontFamily': 'Arial, sans-serif'
    },
    'chat_container': {
        'flexGrow': 1, 'display': 'flex', 'flexDirection': 'column',
        'justifyContent': 'space-between', 'backgroundColor': '#fff',
        'borderRadius': '10px', 'padding': '20px', 'margin': '20px'
    },
    'chat_header': {
        'display': 'flex', 'alignItems': 'center', 'borderBottom': '1px solid #e9ecef', 'paddingBottom': '10px'
    },
    'chat_body': {
        'flexGrow': 1, 'overflowY': 'auto', 'paddingTop': '20px', 'paddingBottom': '20px'
    },
    'chat_bubble': {
        'backgroundColor': '#e9ecef', 'padding': '10px', 'borderRadius': '10px',
        'marginBottom': '10px', 'maxWidth': '60%'
    },
    'input_container': {
        'display': 'flex', 'justifyContent': 'flex-end', 'borderTop': '1px solid #e9ecef', 'paddingTop': '10px'
    },
    'input_box': {
        'padding': '10px', 'border': '1px solid #ced4da',
        'borderRadius': '5px', 'width': '60%'
    },
    'submit_button': {
        'padding': '10px 20px', 'border': 'none', 'backgroundColor': '#007bff',
        'color': 'white', 'borderRadius': '5px', 'marginLeft': '10px', 'cursor': 'pointer'
    }
}

# Layout of chatbot UI
app.layout = html.Div(style=styles['page'], children=[
    # Static Quote at the top
    html.Div(id='quote-display', style={
        'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center',
        'margin': '20px', 'color': '#007bff', 'padding': '10px',
        'border': '1px solid #007bff', 'borderRadius': '5px'
    }, children="Stay Healthy, Stay Happy!"),

    # Chat Window
    html.Div(style=styles['chat_container'], children=[
        # Chat Header
        html.Div(style=styles['chat_header'], children=[
            html.Img(src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR0FRLiXCVFXbf4nUGRtOaBZPRkTuh1AoSrzg', style={'height': '50px', 'borderRadius': '50%'}),
            html.H2("Kuki AI", style={'marginLeft': '20px', 'color': '#343a40'}),
            html.Img(src='https://www.motorindiaonline.in/wp-content/uploads/2016/03/make-in-india-logo.jpg', style={'height': '30px', 'marginLeft': '20px'})
        ]),

        # Chat Body
        html.Div(id='chat-body', style=styles['chat_body'], children=[
            html.Div(style=styles['chat_bubble'], children="Hi there, I'm Kuki 👋"),
            html.Div(style=styles['chat_bubble'], children="I'm a friendly AI, here to chat with you 24/7!")
        ]),

        # Input Field, Dropdown, and Submit Button
        html.Div(style=styles['input_container'], children=[
            dcc.Input(id='input-query', type='text', placeholder='Enter your message...', style=styles['input_box']),
            dcc.Dropdown(
                id='symptom-dropdown',
                options=[{'label': symptom, 'value': symptom} for disease in disease_dict.values() for symptom in disease["Symptoms"]],
                placeholder="Select symptoms",
                multi=True,
                style={'width': '55%', 'marginLeft': '10px', 'marginRight': '10px'}
            ),
            html.Button('Send', id='submit-button', n_clicks=0, style=styles['submit_button'])
        ])
    ])
])

# Callback to update output on query submission
@app.callback(
    Output('chat-body', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('input-query', 'value'), State('symptom-dropdown', 'value'), State('chat-body', 'children')]
)
def update_output(n_clicks, query, selected_symptoms, current_messages):
    # Initialize current_messages if it's None
    if current_messages is None:
        current_messages = []

    if n_clicks > 0:
        user_message = query or ''
        if selected_symptoms:
            user_message += " " + " ".join(selected_symptoms)
        
        # Add user message to chat
        current_messages.append(html.Div(style={**styles['chat_bubble'], 'backgroundColor': '#007bff', 'color': 'white', 'marginLeft': 'auto'}, children=user_message))

        # Check for symptoms in the dictionary
        found_condition = False
        for disease, details in disease_dict.items():
            if any(symptom.lower() in user_message.lower() for symptom in details["Symptoms"]):
                current_messages.append(html.Div(style=styles['chat_bubble'], children=f"You may have a medical condition: {disease}."))
                advice = "\n".join(details["Advice"])
                current_messages.append(html.Div(style=styles['chat_bubble'], children=f"Here is some advice: \n{advice}"))
                found_condition = True
                break
        
        # If no condition is found, use the model
        if not found_condition:
            medical_prediction = medical_symptom_detector(user_message)
            current_messages.append(html.Div(style=styles['chat_bubble'], children=f"You may have a medical condition: {medical_prediction}"))

        return current_messages

    return current_messages

if __name__ == '__main__':
    app.run_server(debug=True, port=2004)
