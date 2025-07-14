#!/bin/bash

# Monitor Final Experiment Progress
# This script provides various ways to monitor the running experiment

echo "Final Experiment Monitor"
echo "======================="

# Check if screen session exists
if screen -list | grep -q "final_experiment"; then
    echo "Screen session 'final_experiment' is running."
    echo ""
    
    # Show log file status
    if [ -f "rizhi.log" ]; then
        echo "Log file status:"
        echo "  File: rizhi.log"
        echo "  Size: $(du -h rizhi.log | cut -f1)"
        echo "  Lines: $(wc -l < rizhi.log)"
        echo "  Last modified: $(stat -c %y rizhi.log)"
        echo ""
        
        echo "Recent log entries (last 20 lines):"
        echo "======================================"
        tail -20 rizhi.log
        echo ""
    else
        echo "Log file 'rizhi.log' not found yet."
        echo ""
    fi
    
    # Show GPU usage
    echo "Current GPU Usage:"
    echo "=================="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
    
    # Show results directory status
    if [ -d "results_final" ]; then
        echo "Results directory status:"
        echo "========================"
        echo "  Directory: results_final"
        echo "  Subdirectories: $(find results_final -type d | wc -l)"
        echo "  Total files: $(find results_final -type f | wc -l)"
        echo "  Total size: $(du -sh results_final | cut -f1)"
        echo ""
        
        # Show latest experiment directories
        echo "Latest experiment directories:"
        ls -lt results_final/ | head -10
        echo ""
    else
        echo "Results directory 'results_final' not found yet."
        echo ""
    fi
    
    echo "Monitoring options:"
    echo "=================="
    echo "  1. Attach to screen session:     screen -r final_experiment"
    echo "  2. Follow log file:              tail -f rizhi.log"
    echo "  3. Watch GPU usage:              watch -n 5 nvidia-smi"
    echo "  4. Monitor results directory:    watch -n 30 'ls -la results_final/'"
    echo "  5. Re-run this monitor:          ./monitor_experiment.sh"
    echo ""
    
else
    echo "Screen session 'final_experiment' is not running."
    echo ""
    echo "To start the experiment:"
    echo "  ./run_final_experiment.sh"
    echo ""
fi 