import subprocess
import sys

# 接收整个命令行参数（除了脚本名称）
params_list = [f'"{p}"' for p in sys.argv[1:]]
params = ' '.join(params_list)

# 构建完整的命令，包括配置文件、设备和参数
command = f"wuji-fit configs/spo2/multi-loss.yaml {params}"

print(command)
# 使用 subprocess.Popen 来执行命令
with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
    # 实时打印输出
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(line, end='')

# 确保进程已结束
proc.wait()

