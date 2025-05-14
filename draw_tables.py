import os
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_name_mapping = {
    "deepseek-reasoner": "DS-R1",
    "gpt-4o-mini": "Gpt4o-Mini",
    "claude-3-7-sonnet-20250219": "Claude3",
    "gemini-2.0-flash": "Gemini2",
    "deepseek-chat": "DS-V3",
    "qwq": "QWQ",
    "qwen3:32b": "QWen3",
    "gemma3:27b": "Gemma3",
    "mistral-small3.1": "Mistral"
}

if __name__ == "__main__":
    
    '''This is to draw table for each protocol to get the matching m,etrics'''
    '''Draaw 14 tables'''
    trans_match_partial_df = pd.read_csv("transitions_match_results_partial.csv")
    trans_match_partial_df['Model'] = trans_match_partial_df['Model'].replace(model_name_mapping)
    protocol_dfs = {protocol: group.reset_index(drop=True) for protocol, group in trans_match_partial_df.groupby("Protocol")}
    print(trans_match_partial_df.head())
    #Generate LaTeX code for each protocol
    output_path = "all_transition_partial_metrics.tex"

    # with open(output_path, "w") as out_f:
    #     # Optional: add a LaTeX document header if you want a standalone .tex
    #     # out_f.write("\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\n\n")

    #     for protocol, df in protocol_dfs.items():
    #         # Generate the LaTeX code for this protocol’s table
    #         latex_code = df[
    #             ['Protocol', 'Model', 'TotalExtracted', 'TotalGT', 'Matched', 'Precision', 'Recall', 'F1-Score']
    #         ].to_latex(
    #             index=False,
    #             float_format="%.3f",
    #             column_format="llcccccc",
    #             longtable=False,
    #             caption=f"{protocol} Partially Correct Transition Extraction Metrics",
    #             label=f"tab:{protocol.lower()}-partial-transition-matching-metrics",
    #             escape=False
    #         )

    #         # Write it out, plus a pagebreak between tables
    #         out_f.write(latex_code)

    #     # Optional: close the document if you added a header
    #     # out_f.write("\\end{document}\n")
        
    # print(f"Wrote all tables into {output_path}")
    
    '''Draw one concludeion table'''
    # Group by Model and calculate the sum for Total Extracted, Total GT, and Matched
    model_summary = trans_match_partial_df.groupby("Model")[["TotalExtracted", "TotalGT", "Matched"]].sum().reset_index()

    # Calculate overall Precision, Recall, and F1-Score for each model
    model_summary["Precision"] = model_summary["Matched"] / model_summary["TotalExtracted"]
    model_summary["Recall"] = model_summary["Matched"] / model_summary["TotalGT"]
    model_summary["F1-Score"] = 2 * (model_summary["Precision"] * model_summary["Recall"]) / (model_summary["Precision"] + model_summary["Recall"])

    latex_code = model_summary.to_latex(
        index=False,
        float_format="%.3f",
        column_format="lcccccc",
        longtable=False,
        caption="Overall Model Performance on Partial Transition Matching of Different Protocols",
        label="tab:partial-transition-match-summary",
        escape=False
    )
    
    
    print("Model performance summary generated successfully.\n", latex_code)