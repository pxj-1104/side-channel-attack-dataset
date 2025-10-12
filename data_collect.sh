# Define interval time and duration
interval=20
duration=11  # Duration (seconds)
test_number=6000 # Starting position
output_base_dir="./test"
attacks_dir="/home/peng/dataset/attacks"

# Hardware events to sample
events=(
    "cache-references"
    "cache-misses"
    "L1-dcache-loads"
    "L1-dcache-load-misses"
    "L1-icache-load-misses"
    "LLC-loads"
    "LLC-load-misses"
    "LLC-store-misses"
)
        # Combine events into a comma-separated string
event_string=$(IFS=,; echo "${events[*]}")

# Create base directories
for load_dir in "low" "medium" "high"; do
    mkdir -p "${output_base_dir}/${load_dir}/normal"
    mkdir -p "${output_base_dir}/${load_dir}/attack"
done


# Get the list of executable files in the attack directory
attack_files=($(ls ${attacks_dir}))


while true; do
      # Ensure the load ratio is 2:5:3
    load_types=("low" "low" "medium" "medium" "medium" "medium" "medium" "high" "high" "high")
    load_type=${load_types[$RANDOM % ${#load_types[@]}]}
    echo "loadtype: $load_type"
    if [ "$load_type" == "low" ]; then
        A=$(shuf -i 1-2 -n 1)
        B=$(shuf -i 100-512 -n 1)
        C=$(shuf -i 1-5 -n 1)
        D=0
        E=0
        F=$(shuf -i 15-30 -n 1)
        G=2
        loadlimit=$(shuf -i 20-35 -n 1)
    elif [ "$load_type" == "medium" ]; then
        A=$(shuf -i 2-4 -n 1)
        B=$(shuf -i 512-1024 -n 1)
        C=$(shuf -i 5-10 -n 1)
        D=1
        E=0
        F=$(shuf -i 31-40 -n 1)
        G=$(shuf -i 2-3 -n 1)
        loadlimit=$((50 - (G- 1) * 10))
    elif [ "$load_type" == "high" ]; then
        A=$(shuf -i 3-6 -n 1)
        B=$(shuf -i 1024-4096 -n 1)
        C=$(shuf -i 10-15 -n 1)
        D=1
        E=1
        F=$(shuf -i 41-60 -n 1)
        G=$(shuf -i 3-4 -n 1)
        loadlimit=$((75 - (G- 2) * 10))
    else
        echo "Invalid load type! Please choose low, medium, or high."
        exit 1
    fi

    # Print the set values
    echo "Running stress-ng with the following parameters:"
    echo "CPU workers (A): $A"
    echo "CPU load (F): $F%"
    echo "VM bytes (B): $B MB"
    echo "HDD bytes (C): $C MB"
    echo "Net devices (D): $D"
    echo "Cache stressors (E): $E"
    echo "Random (G): $G"
    echo "Loadlimit: $loadlimit"
    # Build stress-ng command, here +2 is to ensure the stress-ng execution time is longer than perf, as we have a sleep interval
    cmd="stress-ng --cpu $A --cpu-load $F --vm 1 --vm-bytes ${B}M --vm-keep --vm-hang 1s --hdd 1 --hdd-bytes ${C}M --timeout $((duration + 2))s"
    # If netdev value is not 0, add corresponding parameters
    if [ "$D" -ne 0 ]; then
        cmd="$cmd --netdev $D"
    fi

    # If cache value is not 0, add corresponding parameters
    if [ "$E" -ne 0 ]; then
        cmd="$cmd --cache $E"
    fi

    # Execute stress-ng command
    cmd2="stress-ng --random $G --timeout $((duration + 2))s"

    echo "running $cmd"
    echo "running $cmd2"


    eval "$cmd"  > test.txt &
    eval "$cmd2" > test.txt&

    sleep 0.5

    # Get all stress-ng process PIDs
    pids=$(pgrep stress-ng)

    # Limit CPU usage for each stress-ng process based on the random value for the current load version
    for pid in $pids; do
      cpulimit -p $pid -l $loadlimit > /dev/null 2>&1 &
    done

    # Randomly select whether to execute an attack file
    attack_pid=0
    attack_selected=false
    attack_file=""
    if [ $((RANDOM % 8)) -lt 3 ]; then # The attack ratio is roughly 1:1:1:5
        attack_selected=true
        # Randomly select an attack file
        attack_file=${attack_files[$RANDOM % ${#attack_files[@]}]}
        attack_file_path="${attacks_dir}/${attack_file}"

        # Execute the attack file with a time limit
        timeout ${duration}s "${attack_file_path}" &
        attack_pid=$!

        # Output which attack file was selected
        echo "Selected attack file: $attack_file"
    else
        # Output that no attack file was selected
        echo "No attack file selected"
    fi


    if $attack_selected; then
        output_dir="${output_base_dir}/${load_type}/attack/${attack_file}"
    else
        output_dir="${output_base_dir}/${load_type}/normal"
    fi


    # Create output subdirectory for the current test number
    output_dir="${output_dir}/${test_number}"
    mkdir -p "$output_dir"

    # Collect hardware event data (collecting all events at once)
    sudo perf stat -a -e ${event_string} -I ${interval} sleep ${duration} &> "${output_dir}/hardware_events.txt"

    # Wait for stress test to finish
    wait $stress_pid

    # Wait for all perf processes to finish
    for pid in "${pids[@]}"; do
         if ps -p $pid > /dev/null 2>&1; then
             wait $pid > /dev/null
         fi
    done

    # Terminate attack processes (if any)
    if [ $attack_pid -ne 0 ]; then
        kill -9 $attack_pid > /dev/null 2>&1
    fi

    # Clear the pids array
    pids=()
    echo "collect $test_number finish "
    test_number++
    # Wait for 5 seconds between test cases to ensure previous test has stopped and avoid data overlap
    sleep 5
done
