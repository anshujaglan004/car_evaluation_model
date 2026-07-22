import os
import joblib
import gradio as gr

# ==========================================================
# Load Model
# ==========================================================
try:
    deployed_xgb = joblib.load("car_evaluation_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
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

    if deployed_xgb is None:
        return "❌ Model could not be loaded."

    try:
        input_data = [[
            int(buying_price),
            int(maintenance_cost),
            int(number_of_doors),
            int(number_of_persons),
            int(lug_boot),
            int(safety)
        ]]

        prediction = deployed_xgb.predict(input_data)[0]

        labels = {
            0: "Unacceptable (unacc)",
            1: "Acceptable (acc)",
            2: "Good (good)",
            3: "Very Good (vgood)"
        }

        result = labels.get(int(prediction), str(prediction))

        return f"""
🚗 Car Evaluation Result

Prediction : {result}
"""

    except Exception as e:
        return f"❌ Prediction Error\n\n{e}"


# ==========================================================
# Description
# ==========================================================
DESCRIPTION = """
# 🚙 Car Evaluation Prediction System

This application predicts the acceptability of a car using a trained XGBoost Machine Learning model.

Select all values below and click **Submit**.
"""


# ==========================================================
# Interface
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
            label="Buying Price"
        ),

        gr.Dropdown(
            choices=[
                ("Low", 0),
                ("Medium", 1),
                ("High", 2),
                ("Very High", 3)
            ],
            label="Maintenance Cost"
        ),

        gr.Dropdown(
            choices=[
                ("2", 2),
                ("3", 3),
                ("4", 4),
                ("5 or More", 5)
            ],
            label="Number of Doors"
        ),

        gr.Dropdown(
            choices=[
                ("2 Persons", 2),
                ("4 Persons", 4),
                ("More", 5)
            ],
            label="Number of Persons"
        ),

        gr.Dropdown(
            choices=[
                ("Small", 0),
                ("Medium", 1),
                ("Big", 2)
            ],
            label="Luggage Boot"
        ),

        gr.Dropdown(
            choices=[
                ("Low", 0),
                ("Medium", 1),
                ("High", 2)
            ],
            label="Safety"
        )

    ],

    outputs=gr.Textbox(
        label="Prediction",
        lines=5
    ),

    title="🚙 Car Evaluation System",

    description=DESCRIPTION
)


# ==========================================================
# Launch
# ==========================================================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
