   import joblib

   def test_model_loads():
       model = joblib.load("churn_model.pkl")
       assert model is not None
