import joblib
import pandas as pd
import gradio as gr

# ==============================
# Load Model
# ==============================
try:
    model = joblib.load("car_evaluation_model.pkl")   # Rename your .pkl file to this
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None


# ==============================
# Prediction Function
# ==============================
def predict(
    buying_price,
    maintenance_cost,
    number_of_doors,
    number_of_persons,
    lug_boot,
    safety,
):

    if model is None:
        return "Model not loaded."

    input_data = pd.DataFrame([{
        "buying price": buying_price,
        "maintenance cost": maintenance_cost,
        "number of doors": number_of_doors,
        "number of persons": number_of_persons,
        "lug_boot": lug_boot,
        "safety": safety
    }])

    prediction = model.predict(input_data)[0]

    return f"Predicted Decision: {prediction}"


# ==============================
# Gradio Interface
# ==============================

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Buying Price"),
        gr.Number(label="Maintenance Cost"),
        gr.Number(label="Number of Doors"),
        gr.Number(label="Number of Persons"),
        gr.Number(label="Luggage Boot"),
        gr.Number(label="Safety"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Car Evaluation Prediction",
    description="Enter the encoded numerical values to predict the decision."
)


# ==============================
# Launch
# ==============================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
