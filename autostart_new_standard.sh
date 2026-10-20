sleep 5
cd /home/bb3/ov_develop
screen \
    -L \
    -Logfile logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog \
    bash -c '
        while true; do
            ./build/ovgimbal_standard_mpc configs/standard3.yaml
            exit_code=$?
            echo "[$(date "+%Y-%m-%d %H:%M:%S")] ovgimbal_standard_mpc exited with code ${exit_code}; restarting in 2 seconds..."
            sleep 2
        done
    '
