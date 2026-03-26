import pytest
from app import predict, TextInput

def test_positive_sentiment():
    input_data = TextInput(text="I absolutely loved this movie! Best ever.")
    result = predict(input_data)
    
    assert result["sentiment"] == "POSITIVE"
    assert result["confidence_score"] > 0.6
    assert "confidence" in result

def test_negative_sentiment():
    input_data = TextInput(text="This was the worst film I have ever seen.")
    result = predict(input_data)
    
    assert result["sentiment"] == "NEGATIVE"
    assert result["confidence_score"] > 0.6
    assert "confidence" in result
