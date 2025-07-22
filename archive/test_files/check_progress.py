#!/usr/bin/env python3
"""
Progress Monitor - Check status of running comprehensive experiments
"""

import os
import glob
import time
from datetime import datetime

def check_progress():
    """Check progress of comprehensive experiments"""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE EXPERIMENT PROGRESS CHECK")
    print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check for result files
    results_dir = "results/final_paper_submission_ready"
    if os.path.exists(results_dir):
        print(f"✅ Results directory exists: {results_dir}")
        
        # Check for CSV and JSON files
        csv_files = glob.glob(os.path.join(results_dir, "COMPREHENSIVE_FINAL_RESULTS_*.csv"))
        json_files = glob.glob(os.path.join(results_dir, "COMPREHENSIVE_FINAL_RESULTS_*.json"))
        summary_files = glob.glob(os.path.join(results_dir, "FINAL_EXPERIMENT_SUMMARY_*.md"))
        
        if csv_files:
            latest_csv = max(csv_files, key=os.path.getctime)
            print(f"📄 Latest CSV file: {os.path.basename(latest_csv)}")
            
            # Check file size as progress indicator
            size = os.path.getsize(latest_csv)
            print(f"📏 File size: {size} bytes")
            
            if size > 0:
                try:
                    import pandas as pd
                    df = pd.read_csv(latest_csv)
                    completed = len(df)
                    successful = len(df[df.get('status', 'SUCCESS') != 'FAILED'])
                    failed = len(df[df.get('status', '') == 'FAILED'])
                    
                    print(f"📈 Experiments completed: {completed}/15")
                    print(f"✅ Successful: {successful}")
                    if failed > 0:
                        print(f"❌ Failed: {failed}")
                    
                    if completed > 0:
                        print(f"📊 Progress: {(completed/15)*100:.1f}%")
                        
                        # Show latest results
                        if not df.empty:
                            latest = df.iloc[-1]
                            print(f"🔬 Latest experiment: {latest.get('experiment_name', 'Unknown')}")
                            if 'final_accuracy' in latest:
                                print(f"   Accuracy: {latest['final_accuracy']:.4f}")
                            if 'precision' in latest:
                                print(f"   Precision: {latest['precision']:.4f}")
                except Exception as e:
                    print(f"⚠️ Could not read CSV file: {e}")
        else:
            print("📄 No CSV files found yet")
            
        if summary_files:
            latest_summary = max(summary_files, key=os.path.getctime)
            print(f"📖 Latest summary: {os.path.basename(latest_summary)}")
    else:
        print("⏳ Results directory not created yet - experiments may still be starting")
    
    # Check for training progress images (indicates active training)
    progress_images = glob.glob("training_progress_*.png")
    if progress_images:
        latest_image = max(progress_images, key=os.path.getctime)
        image_time = datetime.fromtimestamp(os.path.getctime(latest_image))
        time_diff = datetime.now() - image_time
        print(f"🖼️ Latest training image: {os.path.basename(latest_image)}")
        print(f"⏱️ Generated: {time_diff.seconds//60} minutes ago")
        
        if time_diff.seconds < 300:  # Less than 5 minutes
            print("🟢 Recent activity detected - experiments likely running")
        else:
            print("🟡 No recent activity - may be between experiments or completed")
    
    # Check for log files
    log_files = glob.glob("logs/*.log")
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"📝 Latest log: {os.path.basename(latest_log)}")
    
    # Check for model weights (indicates training completion)
    weight_files = glob.glob("model_weights/*.pth")
    if weight_files:
        latest_weights = max(weight_files, key=os.path.getctime)
        print(f"🏋️ Latest model weights: {os.path.basename(latest_weights)}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    check_progress() 