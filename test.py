import requests

def get_sentiment(text, api_url):
    """
    Sends a request to the sentiment analysis API and returns the detected emotions.
    """
    payload = {"text": text}  # Adjust based on your API request format
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise an error for HTTP failures (4xx, 5xx)
        emotions = response.json()
        return emotions
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Example usage
api_url = "http://127.0.0.1:8000/sentiment/"  # Replace with your ngrok or deployed API URL
sample_text = "I feel stressed but also hopeful about the future."

emotion_response = get_sentiment(sample_text, api_url)
print("Detected Emotions:", emotion_response)

# Output
# Detected Emotions: {'joy': 0.2, 'sadness': 0.3, 'anger': 0.1, 'fear': 0.4}