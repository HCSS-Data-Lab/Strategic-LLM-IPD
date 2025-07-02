import pandas as pd
import os
import sys
import time
import json
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    from google.generativeai.types.generation_types import StopCandidateException
except ImportError:
    print("Google GenerativeAI library not found. Please run 'pip install google-generativeai'")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Anthropic library not found. Please run 'pip install anthropic'")
    sys.exit(1)

# --- Configuration ---
load_dotenv("Axelrod.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

gemini_model = "gemini-1.5-flash-latest"
anthropic_model = "claude-3-haiku-20240307"

# --- System Prompt for the LLM Coders ---
SYSTEM_PROMPT = """
You are a research assistant performing qualitative coding on text from a Prisoner's Dilemma experiment.
You will be given a 'rationale' written by an AI agent before it made a move.
Your task is to classify this rationale along two dimensions: Horizon Awareness and Opponent Modeling.

You MUST respond ONLY with a valid JSON object with two keys: "horizon_awareness" and "opponent_modeling".

**Dimension 1: Horizon Awareness**
- Measures: Does the text reference the game's length, remaining rounds, or the termination probability?
- Codes:
  - "Explicit": The rationale explicitly mentions a number of rounds, a specific probability, or a direct reference to the end of the game (e.g., "With few turns left...", "10% chance to end").
  - "Implicit": The rationale makes a general reference to the game's length without specifics (e.g., "The game is short", "Since this is a long game").
  - "None": No mention of the game's horizon.

**Dimension 2: Opponent Modeling**
- Measures: Does the agent articulate a hypothesis about the opponent's strategy or type?
- Codes:
  - "Yes": The rationale contains a hypothesis about the opponent's behavior or strategy (e.g., "They seem deterministic", "My opponent is a TitForTat", "They will probably defect").
  - "No": No hypothesis is mentioned.

Example Rationale: "My opponent has cooperated every time, so I think they are a simple TitForTat. The game could end any time, so I will continue to cooperate."
Correct JSON response:
{
  "horizon_awareness": "Implicit",
  "opponent_modeling": "Yes"
}
"""

def get_gemini_coding(rationale_text):
    if not GEMINI_API_KEY:
        return None, "Gemini API key not found"
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(gemini_model)
    
    try:
        response = model.generate_content([SYSTEM_PROMPT, rationale_text])
        return response.text, None
    except StopCandidateException as e:
        return None, f"Gemini API Error (StopCandidateException): {e}"
    except Exception as e:
        return None, f"Gemini API Error: {e}"

def get_anthropic_coding(rationale_text):
    if not ANTHROPIC_API_KEY:
        return None, "Anthropic API key not found"
        
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    try:
        message = client.messages.create(
            model=anthropic_model,
            max_tokens=100,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": rationale_text}]
        )
        return message.content[0].text, None
    except Exception as e:
        return None, f"Anthropic API Error: {e}"

def machine_coder(file_path='labeling_sample.csv'):
    df = pd.read_csv(file_path, keep_default_na=False)
    
    coders = {
        'gemini': {'func': get_gemini_coding, 'ha_col': 'horizon_awareness_gemini', 'om_col': 'opponent_modeling_gemini'},
        'anthropic': {'func': get_anthropic_coding, 'ha_col': 'horizon_awareness_anthropic', 'om_col': 'opponent_modeling_anthropic'}
    }

    # Add columns if they don't exist
    for coder_info in coders.values():
        if coder_info['ha_col'] not in df.columns:
            df[coder_info['ha_col']] = ''
        if coder_info['om_col'] not in df.columns:
            df[coder_info['om_col']] = ''
    
    # Make sure columns are strings
    for coder_info in coders.values():
        df[coder_info['ha_col']] = df[coder_info['ha_col']].astype(str)
        df[coder_info['om_col']] = df[coder_info['om_col']].astype(str)

    total_rows = len(df)
    
    for idx, row in df.iterrows():
        for coder_name, coder_info in coders.items():
            ha_col, om_col = coder_info['ha_col'], coder_info['om_col']
            
            # Check if this rationale is already coded by this coder
            if row[ha_col] or row[om_col]:
                continue
            
            print(f"[{idx+1}/{total_rows}] Coding with {coder_name.capitalize()} for rationale ID {row['rationale_id']}...")
            
            response_text, error = coder_info['func'](row['rationale'])
            
            if error:
                print(f"  - ERROR: {error}")
                df.loc[idx, ha_col] = "API_ERROR"
                df.loc[idx, om_col] = "API_ERROR"
            else:
                try:
                    # Clean the response and parse JSON
                    clean_response = response_text.strip().replace('```json', '').replace('```', '').strip()
                    json_response = json.loads(clean_response)
                    df.loc[idx, ha_col] = json_response.get("horizon_awareness", "PARSE_ERROR")
                    df.loc[idx, om_col] = json_response.get("opponent_modeling", "PARSE_ERROR")
                    print("  - Success.")
                except json.JSONDecodeError:
                    print(f"  - ERROR: Could not parse JSON from response: {response_text[:100]}...")
                    df.loc[idx, ha_col] = "PARSE_ERROR"
                    df.loc[idx, om_col] = "PARSE_ERROR"
            
            # Save after each API call to be safe
            df.to_csv(file_path, index=False)
            time.sleep(1) # Be nice to the APIs

    print("\nMachine coding complete.")

if __name__ == "__main__":
    machine_coder() 