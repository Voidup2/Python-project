from deepface import DeepFace

def recognize_expression(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    
    if isinstance(result, list):
        result = result[0] 
    
    return result.get('emotion', 'unknown')

