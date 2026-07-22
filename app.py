import os
import joblib
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
try:
    deployed_xgb = joblib.load("car_evaluation_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Warning: Model not found or error loading. {e}")
    deployed_xgb = None


# ==========================================================
# Prediction Function
# ==========================================================
def predict_car_safety(
    buying_price,
    maintenance_cost,
    number_of_doors,
    number_of_persons,
    lug_boot,
    safety,
):
    values = [
        buying_price,
        maintenance_cost,
        number_of_doors,
        number_of_persons,
        lug_boot,
        safety,
    ]

    # Check for empty inputs
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please select all input fields."

    # Convert inputs to integers
    try:
        buying_price = int(buying_price)
        maintenance_cost = int(maintenance_cost)
        number_of_doors = int(number_of_doors)
        number_of_persons = int(number_of_persons)
        lug_boot = int(lug_boot)
        safety = int(safety)
    except (ValueError, TypeError):
        return "❌ Invalid input values."

    if deployed_xgb is None:
        return "❌ Model failed to load."

    try:
        input_data = [[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety
        ]]

        prediction = deployed_xgb.predict(input_data)[0]

        # Change this mapping if your label encoding is different
        result_map = {
            0: "Unacceptable (unacc)",
            1: "Acceptable (acc)",
            2: "Good (good)",
            3: "Very Good (vgood)"
        }

        final_result = result_map.get(prediction, str(prediction))

        return f"""
🚗 Car Evaluation Result

Prediction: **{final_result}**
"""

    except Exception as e:
        return f"❌ Prediction Error\n\n{e}"


# ==========================================================
# App Description
# ==========================================================
DESCRIPTION = """
# 🚙 Car Evaluation Prediction System

This application predicts the evaluation of a car using a trained **XGBoost Machine Learning Model**.

Please select all the encoded values below and click **Submit**.
"""


# ==========================================================
# Gradio Interface
# ==========================================================
interface = gr.Interface(
    fn=predict_car_safety,
    inputs=[
        gr.Dropdown(
            choices=[
                ("Low", 0),
                ("Medium", 1),
                ("High", 2),
                ("Very High", 3)
            ],
            label="Buying Price",
        ),
        gr.Dropdown(
            choices=[
                ("Low", 0),
                ("Medium", 1),
                ("High", 2),
                ("Very High", 3)
            ],
            label="Maintenance Cost",
        ),
        gr.Dropdown(
            choices=[
                ("2", 2),
                ("3", 3),
                ("4", 4),
                ("5 or More", 5)
            ],
            label="Number of Doors",
        ),
        gr.Dropdown(
            choices=[
                ("2 Persons", 2),
                ("4 Persons", 4),
                ("More", 5)
            ],
            label="Number of Persons",
        ),
        gr.Dropdown(
            choices=[
                ("Small", 0),
                ("Medium", 1),
                ("Big", 2)
            ],
            label="Luggage Boot",
        ),
        gr.Dropdown(
            choices=[
                ("Low", 0),
                ("Medium", 1),
                ("High", 2)
            ],
            label="Safety",
        ),
    ],
    outputs=gr.Textbox(
        label="Prediction",
        lines=5
    ),
    title="🚙 Car Evaluation System",
    description=DESCRIPTION,
    allow_flagging="never",
)


# ==========================================================
# Launch App
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print(f"Starting server on port {port}...")

    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
