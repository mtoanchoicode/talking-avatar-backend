import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gtts import gTTS
from mutagen.mp3 import MP3
from faster_whisper import WhisperModel
import os, uuid, math
import numpy as np

import faiss
import pickle
import openai
from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app)

# --- OpenAI Setup ---
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Pinecone Setup ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "chat-memory"

# Create index if it doesn't exist (serverless)
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created new Pinecone index: {index_name}")

# Connect to the index
pinecone_index = pc.Index(index_name)
print(f"Connected to Pinecone index: {index_name}")

# --- Globals ---
OUTPUT_DIR = "static/audio"
INDEX_FOLDER = "RAG_index"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Whisper once
whisper_model = WhisperModel("small", device="cpu")

# --- Load FAISS index (raw) ---
index = faiss.read_index("RAG_index/index.faiss")
with open("RAG_index/index.pkl", "rb") as f:
    docstore = pickle.load(f)

# --- Embedding function ---
def embed_text(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def embed_text_pinecone(text: str):
    if not text.strip():
        return [0.0] * 1536
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# --- Retrieve top-k ---
def retrieve(query, k=4):
    q_emb = np.array([embed_text_pinecone(query)], dtype=np.float32)
    D, I = index.search(q_emb, k)

    if len(I) == 0 or I[0][0] == -1:
        print("No relevant context found.")
        return ["No relevant context found."]

    results = []
    for idx in I[0]:
        if idx != -1 and idx in docstore:
            results.append(docstore[idx].page_content)
    return results

# === NEW: Session-based memory retrieval ===
def retrieve_memory(query, session_id, top_k=3):
    q_emb = embed_text_pinecone(query)
    # Filter by session_id
    results = pinecone_index.query(
        vector=q_emb, 
        top_k=top_k, 
        include_metadata=True,
        filter={"session_id": {"$eq": session_id}}
    )
    memories = []
    for match in results.matches:
        if match.score > 0.5:
            memories.append(match.metadata["text"])
    return memories

# === NEW: Save to memory with session_id ===
def save_to_memory(role: str, text: str, session_id: str):
    vector = embed_text(text)
    pinecone_index.upsert([
        (str(uuid.uuid4()), vector, {
            "role": role, 
            "text": text, 
            "session_id": session_id
        })
    ])

# === NEW: Delete session memories ===
def delete_session_memory(session_id: str):
    """Delete all memories for a specific session"""
    try:
        # Query all vectors for this session
        results = pinecone_index.query(
            vector=[0.0] * 1536,  # dummy vector
            top_k=10000,
            include_metadata=True,
            filter={"session_id": {"$eq": session_id}}
        )
        
        # Delete by IDs
        ids_to_delete = [match.id for match in results.matches]
        if ids_to_delete:
            pinecone_index.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} memories for session {session_id}")
    except Exception as e:
        print(f"Error deleting session memory: {e}")

# --- Prompt OpenAI with mode support ---
def ask_openai(query, session_id, use_rag=True):
    if use_rag:
        # RAG Mode: Use knowledge base + memory
        knowledge = retrieve(query)
        knowledge_context = "\n\n".join(knowledge) if knowledge else "No external knowledge found."
        
        memory = retrieve_memory(query, session_id)
        print(f"Retrieved memory for session {session_id}: {memory}")
        memory_context = "\n".join(memory) if memory else ""
        
        system_prompt = f"""You are a friendly 3D avatar assistant.
Use this knowledge when relevant:
{knowledge_context}

Relevant past conversation (if any):
{memory_context}

Answer naturally and warmly. If the user asks about themselves or past chat, use the memory above."""
    else:
        # LLM-only Mode: No RAG, no memory
        system_prompt = """You are a friendly 3D avatar assistant.
Answer naturally and warmly based solely on your training data.
You don't have access to any external knowledge base or conversation history."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# --- Helper: Generate Audio + Blendshapes ---
def generate_tts_and_animation(text: str):
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)

    tts = gTTS(text, lang='en')
    tts.save(filepath)

    duration = MP3(filepath).info.length
    fps = 60
    frame_count = math.ceil(duration * fps)
    frame_time = [i / fps for i in range(frame_count)]

    segments, _ = whisper_model.transcribe(filepath, word_timestamps=True)
    words = []
    for segment in segments:
        for word in segment.words:
            if word.start is not None and word.end is not None:
                words.append({
                    "start": word.start,
                    "end": word.end,
                    "text": word.word.strip()
                })

    blend_data = []
    for t in frame_time:
        active_word = next((w for w in words if w["start"] <= t <= w["end"]), None)

        if active_word:
            w = active_word["text"].lower()
            dur = active_word["end"] - active_word["start"]
            progress = (t - active_word["start"]) / max(dur, 0.001)
            jaw = abs(math.sin(progress * math.pi))

            vowel_map = {
                'a': 0.9, 'e': 0.6, 'i': 0.5, 'o': 0.7, 'u': 0.4
            }
            jaw_scale = next((v for k, v in vowel_map.items() if k in w), 0.3)
            jaw *= jaw_scale

            funnel = 0.5 if 'o' in w or 'oo' in w else 0.0
            pucker = 0.5 if 'u' in w or 'oo' in w else 0.0
        else:
            jaw = funnel = pucker = 0.0

        blink = 0.8 if (int(t * 2) % 180 in range(175, 180)) else 0.0

        blend_data.append({
            "time": round(t, 3),
            "blendshapes": {
                "jawOpen": round(jaw, 3),
                "mouthFunnel": round(funnel, 3),
                "mouthPucker": round(pucker, 3),
                "mouthSmileLeft": round(jaw * 0.3, 3),
                "mouthSmileRight": round(jaw * 0.3, 3),
                "eyeBlinkLeft": round(blink, 3),
                "eyeBlinkRight": round(blink, 3),
                "browInnerUp": 0.1
            }
        })

    return filename, blend_data

@app.route('/talk', methods=['POST'])
def talk():
    try:
        user_input = None
        session_id = None
        use_rag = True

        # Case 1: JSON with 'text'
        if request.is_json:
            data = request.get_json()
            user_input = data.get('text', '').strip()
            session_id = data.get('session_id', 'default')
            use_rag = data.get('use_rag', True)

        # Case 2: Multipart form with 'audio' file
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No audio file'}), 400

            # Get session_id from form
            session_id = request.form.get('session_id', 'default')
            use_rag = request.form.get('use_rag', 'true').lower() == 'true'

            temp_path = os.path.join(OUTPUT_DIR, f"temp_{uuid.uuid4().hex}.wav")
            audio_file.save(temp_path)

            segments, _ = whisper_model.transcribe(temp_path, beam_size=5)
            user_input = " ".join([s.text for s in segments]).strip()

            os.remove(temp_path)

        else:
            return jsonify({'error': 'Send either JSON {text: "..."} or multipart audio file'}), 400

        if not user_input:
            return jsonify({'error': 'Empty input'}), 400

        print(f"Mode: {'RAG' if use_rag else 'LLM-only'}, Session: {session_id}")
        answer = ask_openai(user_input, session_id, use_rag)

        # Save to memory only in RAG mode
        if use_rag:
            save_to_memory("user", user_input, session_id)
            save_to_memory("assistant", answer, session_id)

        filename, blend_data = generate_tts_and_animation(answer)
        print(f"Generated audio file: {filename}")

        return jsonify({
            "filename": f"/static/audio/{filename}",
            "blendData": blend_data
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/new_session', methods=['POST'])
def new_session():
    """Generate a new session ID"""
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear memory for a specific session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'session_id required'}), 400
        
        delete_session_memory(session_id)
        return jsonify({'message': f'Session {session_id} cleared successfully'})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))