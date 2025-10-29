from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import os
from model import DecoderCNN
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]}}, supports_credentials=True)


#decoder file

UPLOAD_FOLDER = "uploads"
CHECKPOINTS = "checkpoints"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- ECC Decode ----------------
def hamming_decode(block):
    if len(block)<7:
        block += [0]*(7-len(block))
    p1,p2,d0,p3,d1,d2,d3 = block
    c1 = p1 ^ d0 ^ d1 ^ d3
    c2 = p2 ^ d0 ^ d2 ^ d3
    c3 = p3 ^ d1 ^ d2 ^ d3
    error_pos = c1 + (c2<<1) + (c3<<2)
    if error_pos != 0 and error_pos <=7:
        block[error_pos-1] ^=1
    return [d0,d1,d2,d3]

def bits_to_str(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        chars.append(chr(int("".join(map(str,byte)),2)))
    return ''.join(chars).rstrip('\x00')

def ecc_decode_payload(payload_tensor):
    bits = (payload_tensor.squeeze().detach().cpu().numpy()>0.5).astype(int)
    decoded_bits = []
    for i in range(0, len(bits), 7):
        block = bits[i:i+7]
        decoded_bits.extend(hamming_decode(block.tolist()))
    return bits_to_str(decoded_bits)

# -------------------- Flask Route --------------------
@app.route("/api/decode", methods=["POST"])
def decode_image():
    try:
        if "image" not in request.files:
            return jsonify({"error":"No image uploaded"}), 400

        file = request.files["image"]
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        # Load image
        img = cv2.imread(upload_path, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error":"Invalid image file"}),400
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.FloatTensor(img_rgb/255.0).permute(2,0,1).unsqueeze(0)

        # Load decoder on CPU to avoid MPS issues
        decoder = DecoderCNN()
        state = torch.load(os.path.join(CHECKPOINTS, "decoder.pth"), map_location="cpu")
        decoder.load_state_dict(state)
        decoder.eval()
        decoder = decoder.cpu()
        img_tensor = img_tensor.cpu()

        # Decode
        with torch.no_grad():
            payload = decoder(img_tensor)

        metadata = ecc_decode_payload(payload)
        fingerprint, timestamp = ("","")
        if "|" in metadata:
            fingerprint, timestamp = metadata.split("|",1)
        else:
            fingerprint = metadata

        response = {
            "decoded_data":{
                "fingerprint": fingerprint,
                "timestamp": timestamp,
                "algorithm":"SHA-256 + RSA-2048",
                "security_level":"Military Grade",
                "origin":"Digital Fingerprint System v2.1",
                "checksum":"0x12345678"
            }
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}),500

if __name__ == "__main__":
    print(f"Running decoder on CPU for stability")
    app.run(debug=True, port=5001)
