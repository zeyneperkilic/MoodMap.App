from flask import Flask, render_template, request, redirect, Response
import requests
import os
import secrets  # Add this import for generating secure secret key

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a secure random secret key

API_URL = "http://localhost:8000"

@app.route('/')
def home():
    return render_template('cover.html')

@app.route('/mood-selection')
def mood_selection():
    return render_template('mood_selection.html')

@app.route('/mood-map/<color>')
def mood_map(color):
    response = requests.get(f"{API_URL}/mood-map/{color}")
    if response.status_code != 200:
        return redirect('/')
    
    data = response.json()
    return render_template(
        "mood_map.html",
        mood_name=data["mood_name"],
        mood_color=color,
        all_songs=data["all_songs"],
        cluster_id=data["cluster_id"]
    )

@app.route('/api/<path:path>', methods=['GET', 'POST'])
def proxy(path):
    if request.method == 'POST':
        response = requests.post(
            f"{API_URL}/{path}",
            json=request.json,
            cookies=request.cookies
        )
    else:
        response = requests.get(
            f"{API_URL}/{path}",
            params=request.args,
            cookies=request.cookies
        )
    
    return Response(
        response.content,
        status=response.status_code,
        content_type=response.headers['content-type']
    )

@app.route('/spotify/login')
def spotify_login():
    return redirect(f"{API_URL}/spotify/login")

@app.route('/logout')
def logout():
    response = requests.post(f"{API_URL}/logout", cookies=request.cookies)
    return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
