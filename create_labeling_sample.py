import os
import csv
import random

def create_sample_for_labeling(directory, output_file, sample_size):
    """
    Creates a random sample of LLM rationales from result files for manual labeling.
    """
    all_rationales = []
    
    files_to_check = [
        'llm_showdown_20250615_171958_consolidated.csv',
        'expanded_evolutionary_pd_20250602_102912_consolidated.csv',
        'expanded_evolutionary_pd_20250602_112311_consolidated.csv',
        'expanded_evolutionary_pd_20250602_150611_consolidated.csv',
        'expanded_evolutionary_pd_20250606_120353_consolidated.csv',
        'expanded_evolutionary_pd_20250607_154031_consolidated.csv',
        'expanded_evolutionary_pd_mutation_20250608_192424_consolidated.csv'
    ]
    
    csv_files = [os.path.join(directory, f) for f in files_to_check]
    
    print("Extracting LLM rationales from all result files...")
    
    for file_path in csv_files:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            in_section = False
            reader = csv.reader(f)
            
            for row in reader:
                if not row:
                    continue
                if "SECTION 3: COMPLETE MATCH AND ROUND DATA" in row[0]:
                    in_section = True
                    header = next(reader)
                    try:
                        agent1_idx = header.index("Agent1")
                        agent2_idx = header.index("Agent2")
                        reasoning1_idx = header.index("Reasoning1")
                        reasoning2_idx = header.index("Reasoning2")
                        # Also grab context
                        round_idx = header.index("Round")
                        move1_idx = header.index("Move1")
                        move2_idx = header.index("Move2")

                    except ValueError:
                        break
                    continue
                
                if "SECTION 4:" in row[0]:
                    in_section = False
                    break

                if in_section:
                    agent1_name = row[agent1_idx]
                    agent2_name = row[agent2_idx]
                    
                    context = (f"File: {os.path.basename(file_path)}, "
                               f"Round: {row[round_idx]}, "
                               f"Agents: {agent1_name} vs {agent2_name}, "
                               f"Moves: ({row[move1_idx]}, {row[move2_idx]})")

                    if "OpenAI" in agent1_name or "Gemini" in agent1_name or "Anthropic" in agent1_name:
                        reasoning = row[reasoning1_idx].strip()
                        if reasoning:
                            all_rationales.append({'rationale': reasoning, 'context': context})
                            
                    if "OpenAI" in agent2_name or "Gemini" in agent2_name or "Anthropic" in agent2_name:
                        reasoning = row[reasoning2_idx].strip()
                        if reasoning:
                            all_rationales.append({'rationale': reasoning, 'context': context})

    print(f"Total rationales extracted: {len(all_rationales)}")
    
    # Randomly sample from the collected rationales
    if len(all_rationales) < sample_size:
        print(f"Warning: Total rationales ({len(all_rationales)}) is less than the desired sample size ({sample_size}). Using all available rationales.")
        sample_size = len(all_rationales)
        
    sampled_rationales = random.sample(all_rationales, sample_size)
    
    print(f"Saving {len(sampled_rationales)} sampled rationales to {output_file}...")

    # Save the sample to a new CSV for labeling
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['rationale_id', 'rationale', 'context', 'horizon_awareness', 'opponent_modeling'])
        writer.writeheader()
        for i, item in enumerate(sampled_rationales):
            writer.writerow({
                'rationale_id': i,
                'rationale': item['rationale'],
                'context': item['context'],
                'horizon_awareness': '', # Placeholder for your label
                'opponent_modeling': ''  # Placeholder for your label
            })
            
    print("Done.")

if __name__ == "__main__":
    results_dir = 'Consolidated results for evo PD'
    output_csv = 'labeling_sample.csv'
    sample_size = 3195
    
    create_sample_for_labeling(results_dir, output_csv, sample_size)
    print(f"\nCreated '{output_csv}'. You can now open this file to begin hand-coding your labels.")
    print("The columns 'horizon_awareness' and 'opponent_modeling' are ready for your input.") 