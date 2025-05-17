from flask import Flask, render_template, request
from gtts import gTTS
import os
import time

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hello_world():
    audio_file = None
    input_value = ""
    if request.method == "POST":
        text = request.form.get("text")
        input_value = text
        if text:
            # Create a unique filename using timestamp
            filename = f"output_{int(time.time())}.mp3"
            audio_path = os.path.join("static", filename)
            
            # Convert text to speech
            tts = gTTS(text=text, lang="bn", slow=False)
            tts.save(audio_path)

            # Return relative path to static folder
            audio_file = f"static/{filename}"

            # Optional: Clean up old audio files to avoid clutter
            for f in os.listdir("static"):
                if f.startswith("output_") and f != filename:
                    os.remove(os.path.join("static", f))

    context = {
            "audio": audio_file, 
            "input_value": input_value
            } 
    
    return render_template("index.html", **context)

if __name__ == "__main__":
    app.run(debug=True)
