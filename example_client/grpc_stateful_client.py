#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import grpc
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray, expand_dims
import classes
import datetime
import argparse
import math
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from client_utils import print_statistics, prepare_certs
from kaldi_python_io import ArchiveReader


def print_debug(msg):
    global debug_mode
    if debug_mode:
        print(msg)


def calculate_utterance_error(referenceArray, resultArray):
    root_mean_err = math.sqrt(
        (np.square(
            resultArray -
            referenceArray)).mean(
            axis=None))
    return root_mean_err


def parse_arguments():
    # Example commands:
    # RM_LSTM4F
    # grpc_stateful_client.py --input_path rm_lstm4f/test_feat_1_10.ark --score_path rm_lstm4f/test_score_1_10.ark
    #     --grpc_address localhost --grpc_port 9000 --input_name Parameter --output_name affinetransform/Fused_Add_
    #     --model_name rm_lstm4f --debug

    # ASPIRE_TDNN
    # grpc_stateful_client.py --input_path aspire_tdnn/mini_feat_1_10.ark,aspire_tdnn/mini_feat_1_10_ivector.ark
    #     --score_path aspire_tdnn/aspire_tdnn_mini_feat_1_10_kaldi_score.ark
    #     --grpc_address localhost --grpc_port 9000 --input_name input,ivector --output_name Final_affine
    #     --model_name aspire_tdnn --cw_l 17 --cw_r 12 --debug

    parser = argparse.ArgumentParser(
        description='Sends requests via TFS gRPC API using data in stateful model ark input file. '
        'It displays performance statistics and average rms errors comparing results with reference scores.'
        'Optionally you can enable debug mode to print additional information about inputs and outputs and detailed accuracy numbers.')
    parser.add_argument(
        '--input_path',
        required=False,
        default='rm_lstm4f/test_feat_1_10.ark',
        help='Path to input ark file')
    parser.add_argument(
        '--score_path',
        required=False,
        default='rm_lstm4f/test_score_1_10.ark',
        help='Path to reference scores ark file')
    parser.add_argument(
        '--grpc_address',
        required=False,
        default='localhost',
        help='Specify url to grpc service. default:localhost')
    parser.add_argument(
        '--grpc_port',
        required=False,
        default=9000,
        help='Specify port to grpc service. default: 9000')
    parser.add_argument(
        '--input_name',
        required=False,
        default='Parameter',
        help='Specify input tensor name. default: Parameter')
    parser.add_argument(
        '--output_name',
        required=False,
        default='affinetransform/Fused_Add_',
        help='Specify output name. default: affinetransform/Fused_Add_')
    parser.add_argument(
        '--model_name',
        default='rm_lstm4f',
        help='Define model name, must be same as is in service. default: rm_lstm4f',
        dest='model_name')

    parser.add_argument(
        '--debug',
        required=False,
        default=0,
        help='Enabling debug prints. Set to 1 to enable debug prints. Default: 0')
    parser.add_argument(
        '--cw_l',
        required=False,
        default=0,
        help='Number of requests for left context window. Works only with context window networks. Default: 0')
    parser.add_argument(
        '--cw_r',
        required=False,
        default=0,
        help='Number of requests for right context window. Works only with context window networks. Default: 0')
    parser.add_argument(
        '--sequence_id',
        required=False,
        default=1,
        help='Sequence ID used by every sequence provided in ARK files. Setting to 0 means sequence will obtain its ID from OVMS. Default: 1')

    print('### Starting grpc_stateful_client.py client ###')

    args = vars(parser.parse_args())
    return args


def prepare_processing_data(args):
    delimiter = ","
    input_files = []
    reference_files = []
    input_names = []
    output_names = []
    input_paths = args.get('input_path').split(delimiter)
    for input_path in input_paths:
        print('Reading input ark file {}'.format(input_path))
        ark_reader = ArchiveReader(input_path)
        input_files.append(ark_reader)

    score_paths = args.get('score_path').split(delimiter)
    for score_path in score_paths:
        print('Reading scores ark file {}'.format(score_path))
        ark_reader = ArchiveReader(score_path)
        reference_files.append(ark_reader)

    inputs = args['input_name'].split(delimiter)
    for input in inputs:
        print('Adding input name {}'.format(input))
        input_names.append(input)

    outputs = args['output_name'].split(delimiter)
    for output in outputs:
        print('Adding output name {}'.format(output))
        output_names.append(output)

    # Validate input
    if len(input_files) != len(input_names):
        print(
            "ERROR: Number of input ark files {} must be equal to the number of input names {}".format(
                len(input_files),
                len(input_names)))
        exit(1)

    if len(reference_files) != len(output_names):
        print(
            "ERROR: Number of output ark files {} must be equal to the number of output names {}".format(
                len(reference_files),
                len(output_names)))
        exit(1)

    # Consolidate input
    input_data = dict()
    name_index = 0
    for ark_reader in input_files:
        input_name = input_names[name_index]
        input_sub_data = dict()
        for name_key, data_obj in ark_reader:
            sequence_size = data_obj.shape[0]
            data = list()
            for input_index in range(0, sequence_size):
                data.append(np.expand_dims(data_obj[input_index], axis=0))
            input_sub_data[name_key] = data
        input_data[input_name] = input_sub_data
        name_index += 1

    # Consolidate reference scores
    name_index = 0
    reference_scores = dict()
    for ark_score in reference_files:
        output_name = output_names[name_index]
        score_ojects = {k: m for k, m in ark_score}
        reference_scores[output_name] = score_ojects
        name_index += 1

    # First ark file is the one we will iterate through for all input ark
    # files per inference request input
    sequence_size_map = dict()
    for sequence_name, data in input_files[0]:
        sequence_size_map[sequence_name] = data.shape[0]
        # Validate reference output scores data for existing sequence names and shapes
        for name in reference_scores:
            score_objects = reference_scores[name]
            if score_objects[sequence_name].shape[0] != data.shape[0]:
                print(
                    "Error reference scores ark file doest not contain proper data for sequence name {0} and data shape {1}".format(
                        sequence_name, data.shape))
                exit(1)

    return sequence_size_map, input_names, output_names, input_data, reference_scores


def validate_output(result, output_names):
    # Validate model output
    for output_name in output_names:
        if output_name not in result.outputs:
            print("ERROR: Invalid output name", output_name)
            print("Available outputs:")
            for o in result.outputs:
                print(o)
            return False
    if 'sequence_id' not in result.outputs:
        print("ERROR: Missing sequence_id in model output")
        print("Available outputs:")
        for o in result.outputs:
            print(o)
        return False
    return True


def main():
    args = parse_arguments()
    global debug_mode
    debug_mode = int(args.get('debug'))

    channel = grpc.insecure_channel(
        "{}:{}".format(
            args['grpc_address'],
            args['grpc_port']))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    processing_times = np.zeros((0), int)
    cw_l = int(args.get('cw_l'))
    cw_r = int(args.get('cw_r'))
    print('Context window left width cw_l: {}'.format(cw_l))
    print('Context window right width cw_r: {}'.format(cw_r))
    get_sequence_id = False
    sequence_id = np.uint64(args.get('sequence_id'))
    if sequence_id == 0:
        get_sequence_id = True
    print('Starting sequence_id: {}'.format(sequence_id))
    print('Start processing:')
    print('Model name: {}'.format(args.get('model_name')))

    sequence_size_map, input_names, output_names, input_data, reference_scores = prepare_processing_data(
        args)

    # Input control tokens
    SEQUENCE_START = 1
    SEQUENCE_END = 2

    global_avg_rms_error_sum = 0.0

    # Input shape information
    for input_name in input_names:
        input_sub_data = input_data[input_name]
        tensor_data = input_sub_data[list(sequence_size_map.keys())[0]][0]
        print_debug('\tInput {} in shape: {}'.format(input_name, tensor_data.shape))

    # Output shape information
    for output_name in output_names:
        score_data = reference_scores[output_name][list(sequence_size_map.keys())[0]][0]
        score_data = np.expand_dims(score_data, axis=0)
        print_debug('\tOutput {} in shape: {}'.format(output_name, score_data.shape))

    # Main inference loop
    for sequence_name, sequence_size in sequence_size_map.items():
        print('\n\tSequence name: {}'.format(sequence_name))
        print('\tSequence size: {}'.format(sequence_size))
        print('\tSequence id: {}'.format(sequence_id))

        if sequence_size == 1:
            print('\nERROR: Detected sequence with only one frame. Every sequence must contain at least 2 frames.'.format(
                sequence_size))
            exit(1)

        mean_avg_rms_error_sum = 0.0
        score_index = (cw_l + cw_r) * -1

        for x in range(0, sequence_size + cw_l + cw_r):
            print_debug('\tExecution: {}\n'.format(x))
            request = predict_pb2.PredictRequest()
            request.model_spec.name = args.get('model_name')

            # Set input data index
            input_index = x
            # Input for context window
            if x < cw_l:
                # Repeating first frame infer request cw_l times for the
                # context window model start
                input_index = 0
            elif x >= cw_l and x < sequence_size + cw_l:
                # Standard processing
                input_index = x - cw_l
            else:
                # Repeating last frame infer request cw_r times for the
                # context window model end
                input_index = sequence_size - 1

            # Setting request input
            for input_name in input_names:
                input_sub_data = input_data[input_name]
                tensor_data = input_sub_data[sequence_name][input_index]
                request.inputs[input_name].CopyFrom(
                    make_tensor_proto(tensor_data, shape=tensor_data.shape))

            # Add sequence start
            if x == 0:
                request.inputs['sequence_control_input'].CopyFrom(
                    make_tensor_proto([SEQUENCE_START], dtype="uint32"))

            # Set sequence id
            request.inputs['sequence_id'].CopyFrom(
                make_tensor_proto([sequence_id], dtype="uint64"))

            # Add sequence end
            if x == sequence_size + cw_l + cw_r - 1:
                request.inputs['sequence_control_input'].CopyFrom(
                    make_tensor_proto([SEQUENCE_END], dtype="uint32"))

            start_time = datetime.datetime.now()
            # result includes a dictionary with all model outputs
            result = stub.Predict(request, 10.0)
            end_time = datetime.datetime.now()

            if not validate_output(result, output_names):
                print(
                    "ERROR: Model result validation error. Adding end sequence inference request for the model and exiting.")
                request.inputs['sequence_control_input'].CopyFrom(
                    make_tensor_proto([SEQUENCE_END], dtype="uint32"))
                result = stub.Predict(request, 10.0)
                exit(1)

            # Unique sequence_id provided by OVMS
            if get_sequence_id:
                sequence_id = np.uint64(result.outputs['sequence_id'])
                get_sequence_id = False

            duration = (end_time - start_time).total_seconds() * 1000
            processing_times = np.append(
                processing_times, np.array([int(duration)]))

            # Compare results after we are pass initial context window results
            if score_index >= 0:
                # Loop over reference output results
                avg_rms_error_sum = 0.0

                for output_name in output_names:
                    score_data = reference_scores[output_name][sequence_name][score_index]

                    # Parse output
                    results_array = make_ndarray(result.outputs[output_name])

                    # Calculate error
                    avg_rms_error = calculate_utterance_error(
                        score_data, results_array[0])
                    avg_rms_error_sum += avg_rms_error

                    # Statistics
                    print_debug(
                        'Output name: {} Rms error: {:.10f}\n'.format(
                            output_name, avg_rms_error))

                mean_avg_rms_error_sum += avg_rms_error_sum
                # Statistics
                print_debug(
                    'Iteration {}; Average rms error: {:.10f} Processing time: {:.2f} ms; speed {:.2f} fps\n'.format(
                        x, avg_rms_error_sum, round(
                            np.average(duration), 2), round(
                            sequence_size / np.average(duration), 2)))
                # END output names loop

            score_index += 1
            # END utterance loop

        seq_avg_rms_error_sum = mean_avg_rms_error_sum / (sequence_size)
        print(
            "\tSequence id: {} ; Sequence name: {} ; Average RMS Error: {:.10f}\n".format(
                sequence_id,
                sequence_name,
                seq_avg_rms_error_sum))
        global_avg_rms_error_sum += seq_avg_rms_error_sum
        # END input name loop

    final_avg_rms_error_sum = global_avg_rms_error_sum / len(sequence_size_map)

    print("Global average rms error: {:.10f}\n".format(
        final_avg_rms_error_sum))
    print_statistics(processing_times, sequence_size / 1000)
    print('### Finished grpc_stateful_client.py client processing ###')


if __name__ == "__main__":
    main()
