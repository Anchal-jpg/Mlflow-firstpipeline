#Import testClient so we can send simulated HTTP requests to our FastAPI application
from fastapi.testclient import TestClient

#Import application object defined in main.py
from main import app

#Reusable test client for all test functions

client = TestClient(app)


#function name should be started with test_ so that pytest can discover it
def test_health_endpoint():
    """Verify that the application is running aand can respond to a health check request"""

    response = client.get("/health")
    assert response.status_code == 200
    #to reduce the if else use assert statement to check the response body
    assert response.json() == {"status": "healthy"}

def test_valid_iris_prediction():
    """Verify that the application can return a valid prediiction"""

    #Define a valid request body
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    #Send a simulated POSt request with JSON input
    response = client.post("predict",json=payload)

    #Endpoint should accept the valid input
    assert response.status_code==200

    result=response.json()
    assert "prediction" in result
    assert "class_name" in result

    assert result["class_name"] in["setosa","versicolor","virginica"]
    assert result["prediction"] in [0,1,2]
