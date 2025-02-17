from deepface import DeepFace

def recognize_expression(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    
    # DeepFace returns a list when multiple faces are detected
    if isinstance(result, list):
        result = result[0]  # Pick the first detected face
    
    return result.get('emotion', 'unknown')

