import os
import openai

# Utility to call OpenAI for recipe validation

def ai_validate_recipe(recipe_title, ingredients, steps, api_key=None):
    """
    Use OpenAI GPT to validate a recipe. Returns (approved: bool, feedback: str, confidence: float)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not set in environment.")
    openai.api_key = api_key

    prompt = f"""
You are a professional chef and food safety expert. Review the following recipe for completeness, clarity, and safety.
If the recipe is clear, complete, and safe, approve it. Otherwise, reject and provide feedback.

Recipe Title: {recipe_title}
Ingredients: {ingredients}
Steps: {steps}

Respond ONLY in JSON with these keys:
    - approved (true/false): Is the recipe approved?
    - confidence (float, 0.0-1.0): Your confidence in the approval decision.
    - feedback (string): Feedback for the trainer.
Example:
{{"approved": true, "confidence": 0.92, "feedback": "Recipe is clear and safe."}}
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a recipe validation assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.2,
    )
    import json
    try:
        content = response.choices[0].message.content
        result = json.loads(content)
        # Use 'confidence' field, fallback to 0.0 if missing
        confidence = float(result.get("confidence", 0.0))
        feedback = str(result.get("feedback", "No feedback provided."))
        # Use 'approved' field directly if present, else fallback to confidence > 0.9
        approved = bool(result.get("approved", confidence > 0.9))
        if approved:
            feedback = feedback + "\nRecipe approved: confidence greater than 90%."
        else:
            feedback = feedback + "\nRecipe rejected: confidence 90% or less. Please address the feedback above."
        return approved, feedback, confidence
    except Exception as e:
        return False, f"AI validation failed: {e}", 0.0