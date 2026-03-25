import pytest
from app import predict, TextInput   # ← this matches your exact repo

def test_positive_sentiment():
    """One function = one purpose. Tests a clear positive case."""
    input_data = TextInput(text="I absolutely loved this movie! Best ever.")
    result = predict(input_data)        # calls your exact live predict function
    
    assert result["sentiment"] == "POSITIVE"
    assert result["confidence_score"] > 0.6   # your BERT model confidence
    assert "confidence" in result             # the string version you return

def test_negative_sentiment():
    """Handles the opposite case. Same rules."""
    input_data = TextInput(text="This was the worst film I have ever seen.")
    result = predict(input_data)
    
    assert result["sentiment"] == "NEGATIVE"
    assert result["confidence_score"] > 0.6
    assert "confidence" in result
