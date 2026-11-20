#!/bin/bash
#SBATCH --job-name=beemonitor_cvpr
#SBATCH --partition=standard
#SBATCH --account=cmg25_cr_default
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16  # Request 8 CPU cores
#SBATCH --time=72:00:00
#SBATCH --output=logs/beemonitor-%j.out
#SBATCH --error=logs/beemonitor-%j.err

# ============================================================================
# BeeMonitor - CVPR Evaluation Video Processing
# ============================================================================

echo "=========================================="
echo "BeeMonitor - CVPR Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

# Create logs directory
mkdir -p logs

# Load environment
module load anaconda
conda activate beemonitor

# Set environment
export PYTHONUNBUFFERED=1

# Navigate to project
cd /storage/work/eai6/eai6/work/eai6/BeeMonitor

echo "Working directory: $(pwd)"
echo ""
echo "Input:  /storage/group/cmg25/default/osmia_data"
echo "Output: /storage/group/cmg25/default/osmia_data_processing_output/"
echo "Mode:   Two-mode tracking (adaptive YOLO ONLY)"
echo ""

# Count videos
VIDEO_COUNT=$(find /storage/group/cmg25/default/osmia_data -name "*.mp4" -o -name "*.MP4" -o -name "*.avi" | wc -l)
echo "Found $VIDEO_COUNT video(s)"
echo ""

# ============================================================================
# RUN PROCESSING
# ============================================================================

echo "Starting analysis..."
echo ""

python cloud_scripts/batch_process.py

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Show results
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Results saved to:"
    echo "  /storage/group/cmg25/default/osmia_data_processing_output/"
    echo ""
    
    TRACKING_COUNT=$(find /storage/group/cmg25/default/osmia_data_processing_output -name "*_tracking.csv" | wc -l)
    EVENTS_COUNT=$(find /storage/group/cmg25/default/osmia_data_processing_output -name "*_events.csv" | wc -l)
    VIDEO_COUNT=$(find /storage/group/cmg25/default/osmia_data_processing_output -name "*_annotated.mp4" | wc -l)
    
    echo "Generated files:"
    echo "  Tracking CSVs:      $TRACKING_COUNT"
    echo "  Events CSVs:        $EVENTS_COUNT"
    echo "  Annotated videos:   $VIDEO_COUNT"
    
    echo ""
    echo "File listing:"
    ls -1h /storage/group/cmg25/default/osmia_data_processing_output/ | head -20
    
    echo ""
    echo "Total file sizes:"
    du -sh /storage/group/cmg25/default/osmia_data_processing_output/
else
    echo ""
    echo "✗ Processing failed. Check logs for errors."
    echo ""
    echo "Last 20 lines of error log:"
    cat logs/beemonitor-$SLURM_JOB_ID.err               #tail -20 logs/beemonitor-$SLURM_JOB_ID.err
fi

exit $EXIT_CODE