#!/bin/bash
set -xe


# tensorflow models pb directly.
function main {
    # import common funcs
    source common.sh
    model_list_json="models.json"

    # set common info
    init_params $@
    fetch_cpu_info
    set_environment

    # requirements
    pip install -r requirements.txt

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # model details
        set_extra_params
        #
        for batch_size in ${batch_size_list[@]}
        do
            logs_path_clean
            generate_core
            # launch
            echo -e "\n\n\n\n Running..."
            cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

function set_extra_params {
    extra_params=" "
    # ckpt
    extra_value="$(jq --arg m ${model_name} '.[$m].output_name' ${model_list_json} |sed 's/"//g')"
    if [ "${extra_value}" != "null" ];then
        extra_params+=" --output_name ${extra_value} "
    fi
    # input output defined
    extra_value="$(jq --arg m ${model_name} '.[$m].input_output_def' ${model_list_json} |sed 's/"//g')"
    if [ "${extra_value}" == "in_model_details" ];then
        extra_params+=" --model_name ${model_name} "
    fi
    # disable optimize
    extra_value="$(jq --arg m ${model_name} '.[$m].disable_optimize' ${model_list_json} |sed 's/"//g')"
    if [ "${extra_value}" == "true" ];then
        extra_params+=" --disable_optimize "
    fi
    # graph from inc for ckpt, saved model and other non pb
    extra_value="$(jq --arg m ${model_name} '.[$m].load_graph_via_inc' ${model_list_json} |sed 's/"//g')"
    if [ "${extra_value}" == "true" ];then
        extra_params+=" --use_nc "
    fi
    # model path
    if [ "${model_path}" == "" ];then
        model_root_path="/home2/tensorflow-broad-product/oob_tf_models"
        extra_value="$(jq --arg m ${model_name} '.[$m].model_path' ${model_list_json} |sed 's/"//g')"
        model_path="${model_root_path}/${extra_value}"
        # quantize model for int8 via INC
        if [ "${precision}" == "int8" ];then
            quantize_int8_model
        fi
    fi
}

function quantize_int8_model {
    # pb saved dir
    int8_saved_dir="${HOME}/int8_pb_from_inc/${OOB_CONDA_ENV}"
    mkdir -p ${int8_saved_dir}
    output_path="${int8_saved_dir}/${framework}-${model_name}-tune.pb"

    # quantize
    if [ "${mode_name}" == "tune" ];then
        rm -rf ${output_path}
        # specifical bs when tuning
        tune_extra_value="$(jq --arg m ${model_name} '.[$m].inc_tune_bs' ${model_list_json} |sed 's/"//g')"
        if [ "${tune_extra_value}" != "null" ];then
            tune_extra_params=" -b ${tune_extra_value} "
        else
            tune_extra_params=" "
        fi
        # status
        status_check_dir="${WORKSPACE}/inc-tune-status"
        mkdir -p ${status_check_dir}
        status_check_log="${status_check_dir}/tune.log"
        # convert
        tune_start_time=$(date +%s)
        tune_return_value=$(
            python tf_benchmark.py \
                    --tune ${addtion_options} ${extra_params} ${tune_extra_params} \
                    --model_path ${model_path} --output_path ${output_path} \
                    > ${status_check_log} 2>&1 && echo $? || echo $?
        )
        tune_end_time=$(date +%s)
        tune_time=$(
            echo |awk -v tune_start_time=$tune_start_time -v tune_end_time=$tune_end_time '{
                tune_time = tune_end_time - tune_start_time;
                print tune_time;
            }'
        )
        if [ "${tune_return_value}" == "0" ];then
            tune_status='SUCCESS'
        else
            tune_status='FAILURE'
            tail ${status_check_log}
        fi
        status_saved_log="${status_check_dir}/${framework}-${model_name}-${tune_status}-${tune_time}.log"
        mv ${status_check_log} ${status_saved_log}
        # only quantize, no need for benchmark
        artifact_url="${BUILD_URL}artifact/inc-tune-status/$(basename ${status_saved_log})"
        echo "${model_name},${tune_status},${tune_time},${artifact_url}" |tee -a ${WORKSPACE}/summary.log
        exit 0
    elif [ ! -e ${output_path} ];then
        # specifical bs when tuning
        tune_extra_value="$(jq --arg m ${model_name} '.[$m].inc_tune_bs' ${model_list_json} |sed 's/"//g')"
        if [ "${tune_extra_value}" != "null" ];then
            tune_extra_params=" -b ${tune_extra_value} "
        else
            tune_extra_params=" "
        fi
        # status
        status_check_dir="${WORKSPACE}/inc-tune-status"
        mkdir -p ${status_check_dir}
        status_check_log="${status_check_dir}/tune.log"
        # convert
        tune_start_time=$(date +%s)
        tune_return_value=$(
            python tf_benchmark.py \
                    --tune ${addtion_options} ${extra_params} ${tune_extra_params} \
                    --model_path ${model_path} --output_path ${output_path} \
                    > ${status_check_log} 2>&1 && echo $? || echo $?
        )
        tune_end_time=$(date +%s)
        tune_time=$(
            echo |awk -v tune_start_time=$tune_start_time -v tune_end_time=$tune_end_time '{
                tune_time = tune_end_time - tune_start_time;
                print tune_time;
            }'
        )
        if [ "${tune_return_value}" == "0" ];then
            tune_status='SUCCESS'
        else
            tune_status='FAILURE'
            tail ${status_check_log}
        fi
        status_saved_log="${status_check_dir}/${framework}-${model_name}-${tune_status}-${tune_time}.log"
        mv ${status_check_log} ${status_saved_log}
        # only quantize, no need for benchmark
        artifact_url="${BUILD_URL}artifact/inc-tune-status/$(basename ${status_saved_log})"
        echo "${model_name},${tune_status},${tune_time},${artifact_url}" |tee -a ${WORKSPACE}/summary.log
    fi
    # return model path for benchmark
    model_path="${output_path}"
    # if tune failed exit
    if [ "${tune_status}" == "FAILURE" ];then
        exit 1
    fi
}

function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"
        # instances
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${OOB_EXEC_HEADER} \
            python tf_benchmark.py --benchmark \
                --model_path ${model_path} \
                --precision ${precision} \
                --batch_size ${batch_size} \
                --num_warmup ${num_warmup} \
                --num_iter ${num_iter} \
                ${extra_params} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
wget -q -O common.sh https://raw.githubusercontent.com/mengfei25/oob-common/main/common.sh

# Start
main "$@"
