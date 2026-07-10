import os
import pandas as pd
import matplotlib.pyplot as plt
from config import CYCLE_LOG_PATH

def calculate_congestion_reduction(fixed_value, adaptive_value):
    """
    Calculates the percentage reduction in congestion/waiting time.
    Formula: ((Fixed - Adaptive) / Fixed) * 100
    """
    if fixed_value == 0:
        return 0.0
    return ((fixed_value - adaptive_value) / fixed_value) * 100

def aggregate_metrics(cycle_log_path):
    """
    Reads the CSV log and calculates average metrics for comparison.
    """
    try:
        df = pd.read_csv(cycle_log_path)
        
        # Checking if the dataframe is empty
        if df.empty:
            print("Warning: Log file is empty.")
            return None
            
        # Summary calculations (we will group these by strategy later)
        summary = {
            "avg_waiting_time": df["waiting_time"].mean(),
            "avg_queue_length": df["queue_length"].mean(),
            "total_throughput": df["throughput"].sum()
        }
        return summary
        
    except FileNotFoundError:
        print(f"Error: {cycle_log_path} not found. Run the simulation first.")
        return None
    
def generate_comparison_charts(cycle_log_path):
    """
    Reads the log file, aggregates performance data by strategy, 
    and generates comparison charts using Matplotlib.
    """
    try:
        df = pd.read_csv(cycle_log_path)
        if df.empty:
            print("No data available to plot.")
            return
        
        # Group by strategy to get average waiting times for the bar chart
        strategy_summary = df.groupby("strategy")["waiting_time"].mean().reset_index()

        # Create a figure with 2 subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Bar Chart: Average Waiting Time per Strategy
        ax1.bar(strategy_summary["strategy"], strategy_summary["waiting_time"], color=['gray', 'blue', 'green'])
        ax1.set_title("Average Waiting Time Comparison")
        ax1.set_xlabel("Signal Strategy")
        ax1.set_ylabel("Time (seconds)")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # 2. Line Chart: Queue Length Over Time/Ticks
        if "tick" in df.columns:
            for strategy in df["strategy"].unique():
                strat_df = df[df["strategy"] == strategy]
                ax2.plot(strat_df["tick"], strat_df["queue_length"], label=strategy, alpha=0.8)
            ax2.set_title("Queue Length Over Time")
            ax2.set_xlabel("Simulation Tick")
            ax2.set_ylabel("Queue Length (vehicles)")
            ax2.legend()
            ax2.grid(linestyle='--', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, "No 'tick' column found\nfor line chart.", 
                     ha='center', va='center', fontsize=12, color='red')
            ax2.set_title("Queue Length Over Time")

        plt.tight_layout()
        
        # Save the visualization as an image
        #plt.savefig("comparison_chart.png")
        #print("Success: charts saved as 'comparison_chart.png'")
        plt.close()

    except FileNotFoundError:
        print(f"Error: Cannot generate charts. {cycle_log_path} not found.")

def generate_summary_report(cycle_log_path, output_summary_path="strategy_summary_report.csv"):
    """
    Groups the metrics by strategy and exports a side-by-side 
    comparison CSV file for final reporting.
    """
    try:
        df = pd.read_csv(cycle_log_path)
        if df.empty:
            print("No data available to generate a summary report.")
            return

        # Calculate means for each strategy
        summary = df.groupby("strategy").agg(
            Avg_Waiting_Time=("waiting_time", "mean"),
            Avg_Queue_Length=("queue_length", "mean"),
            Total_Throughput=("throughput", "sum")
        ).reset_index()

        # Calculate congestion reduction compared to the 'fixed_time' baseline
        fixed_df = summary[summary["strategy"] == "fixed_time"]
        
        if not fixed_df.empty:
            fixed_wait = fixed_df["Avg_Waiting_Time"].values[0]
            
            # Apply our calculation across all strategies
            summary["Congestion_Reduction_Pct"] = summary["Avg_Waiting_Time"].apply(
                lambda x: calculate_congestion_reduction(fixed_wait, x)
            )
        else:
            summary["Congestion_Reduction_Pct"] = 0.0

        # Transpose/pivot the table so strategies sit side-by-side as columns
        summary_side_by_side = summary.set_index("strategy").T

        # Save to a new CSV file
        summary_side_by_side.to_csv(output_summary_path)
        print(f"Success: Summary report saved as '{output_summary_path}'")
        
    except FileNotFoundError:
        print(f"Error: Cannot generate summary. {cycle_log_path} not found.")


# ── Main Execution Block (Now moved properly to the bottom) ──────────────────
if __name__ == "__main__":
    print("--- Running Performance Evaluator ---")
    
    # Check if the simulation log exists before running
    if os.path.exists(CYCLE_LOG_PATH):
        # 1. Run metrics aggregation
        metrics = aggregate_metrics(CYCLE_LOG_PATH)
        print("Aggregated Metrics Summary:", metrics)
        
        # 2. Generate the Matplotlib charts
        generate_comparison_charts(CYCLE_LOG_PATH)
        
        # 3. Generate the side-by-side CSV report
        generate_summary_report(CYCLE_LOG_PATH)
    else:
        print(f"[{CYCLE_LOG_PATH}] not found.")
        print("Please run the main traffic simulation first to generate data logs!")