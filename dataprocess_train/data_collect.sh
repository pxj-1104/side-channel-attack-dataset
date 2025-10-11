# 定义间隔时间和持续时间
interval=20
duration=11  # 持续时间（秒）
test_number=6000 #起始位置
output_base_dir="./test"
attacks_dir="/home/peng/dataset/attacks"

# 待采样硬件事件
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
        # 将事件组合成逗号分隔的字符串
event_string=$(IFS=,; echo "${events[*]}")

# 创建基准目录
for load_dir in "low" "medium" "high"; do
    mkdir -p "${output_base_dir}/${load_dir}/normal"
    mkdir -p "${output_base_dir}/${load_dir}/attack"
done


# 获取攻击目录下的可执行文件列表
attack_files=($(ls ${attacks_dir}))


while true; do
      # 确保负载比为2:5:3
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
        A=$(shuf -i 1-2 -n 1)
        B=$(shuf -i 512-1024 -n 1)
        C=$(shuf -i 5-10 -n 1)
        D=1
        E=0
        F=$(shuf -i 31-40 -n 1)
        G=$(shuf -i 2-3 -n 1)
        loadlimit=$((50 - (G- 1) * 10))
    elif [ "$load_type" == "high" ]; then
        A=$(shuf -i 1-2 -n 1)
        B=$(shuf -i 1024-4096 -n 1)
        C=$(shuf -i 10-15 -n 1)
        D=1
        E=1
        F=$(shuf -i 41-60 -n 1)
        G=$(shuf -i 3-4 -n 1)
        loadlimit=$((75 - (G- 2) * 10))
    else
        echo "无效的负载类型！请选择 low, medium, 或 high。"
        exit 1
    fi

    # 打印设定的值
    echo "Running stress-ng with the following parameters:"
    echo "CPU workers (A): $A"
    echo "CPU load (F): $F%"
    echo "VM bytes (B): $B MB"
    echo "HDD bytes (C): $C MB"
    echo "Net devices (D): $D"
    echo "Cache stressors (E): $E"
    echo "Random (G): $G"
    echo "Loadlimit: $loadlimit"
    # 构建 stress-ng 命令,这里+2是为了保证stress-ng执行时间比perf长，由于后续有休眠作为间隔，因此不担心其执行时间过长
    cmd="stress-ng --cpu $A --cpu-load $F --vm 1 --vm-bytes ${B}M --vm-keep --vm-hang 1s --hdd 1 --hdd-bytes ${C}M --timeout $((duration + 2))s"
    # 如果 netdev 的值不为 0，则添加对应参数
    if [ "$D" -ne 0 ]; then
        cmd="$cmd --netdev $D"
    fi

    # 如果 cache 的值不为 0，则添加对应参数
    if [ "$E" -ne 0 ]; then
        cmd="$cmd --cache $E"
    fi

    # 执行 stress-ng 命令
    cmd2="stress-ng --random $G --timeout $((duration + 2))s"

    echo "running $cmd"
    echo "running $cmd2"


    eval "$cmd"  > test.txt &
    eval "$cmd2" > test.txt&

    sleep 0.5

    # 获取所有 stress-ng 进程的 PID
    pids=$(pgrep stress-ng)

    # 限制每个 stress-ng 进程的 CPU 占用率为根据当前负载版本的 随机值%
    for pid in $pids; do
      cpulimit -p $pid -l $loadlimit > /dev/null 2>&1 &
    done

    # 随机选择是否执行攻击文件
    attack_pid=0
    attack_selected=false
    attack_file=""
    if [ $((RANDOM % 6)) -lt 3 ]; then #攻击比例基本为1:1:1:3
        attack_selected=true
        # 随机选择一个攻击文件
        attack_file=${attack_files[$RANDOM % ${#attack_files[@]}]}
        attack_file_path="${attacks_dir}/${attack_file}"

        # 执行攻击文件，设置执行时间限制
        timeout ${duration}s "${attack_file_path}" &
        attack_pid=$!

        # 输出选择了哪个攻击文件
        echo "Selected attack file: $attack_file"
    else
        # 输出没有选择攻击文件
        echo "No attack file selected"
    fi


    if $attack_selected; then
        output_dir="${output_base_dir}/${load_type}/attack/${attack_file}"
    else
        output_dir="${output_base_dir}/${load_type}/normal"
    fi


    # 为当前测试编号创建输出子目录
    output_dir="${output_dir}/${test_number}"
    mkdir -p "$output_dir"

    # 收集硬件事件数据（所有事件一次性收集）
    sudo perf stat -a -e ${event_string} -I ${interval} sleep ${duration} &> "${output_dir}/hardware_events.txt"

    # 等待压力测试结束
    wait $stress_pid

    # 等待所有 perf 进程结束
    for pid in "${pids[@]}"; do
         if ps -p $pid > /dev/null 2>&1; then
             wait $pid > /dev/null
             #echo "Skipping wait for PID $pid: not a child process or already terminated."
         fi
    done

    # 终止攻击进程（如果有）
    if [ $attack_pid -ne 0 ]; then
        kill -9 $attack_pid > /dev/null 2>&1
    fi

    # 清空 pids 数组
    pids=()
    echo "collect $test_number finish "
    test_number++
    # 每个测试案例之间休息 5秒,保证停止上一次的测试，避免数据重叠
    sleep 5
done









