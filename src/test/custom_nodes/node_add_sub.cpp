//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../custom_node_interface.hpp"

extern "C" int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    if (paramsLength != 2) {
        return 1;
    }

    if (inputsLength != 1) {
        return 2;
    }

    if (strcmp(inputs[0].name, "input_numbers") != 0) {
        return 3;
    }

    const struct CustomNodeTensor* input = &inputs[0];

    if (input->precision != CustomNodeTensorPrecision::FP32) {
        return 4;
    }

    float addValue = 0.0f;
    float subValue = 0.0f;

    for (int i = 0; i < paramsLength; i++) {
        if (strcmp(params[i].key, "add_value") == 0) {
            addValue = atof(params[i].value);
        }
        if (strcmp(params[i].key, "sub_value") == 0) {
            subValue = atof(params[i].value);
        }
        // precision
        // int => width
    }

    printf("CUSTOM ADD_SUB NODE => Parameters passed: add_value:(%f); sub_value:(%f)\n", addValue, subValue);
    printf("CUSTOM ADD_SUB NODE => Number of input tensors passed: (%d)\n", inputsLength);

    for (int i = 0; i < inputsLength; i++) {
        printf("CUSTOM ADD_SUB NODE => Input Name(%s) DataLen(%ld) DimLen(%ld)\n", inputs[i].name, inputs[i].dataLength, inputs[i].dimsLength);
    }

    *outputsLength = 1;
    *outputs = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor) * (*outputsLength));
    struct CustomNodeTensor* output = (&(*outputs))[0];

    output->name = "output_numbers";
    output->data = (uint8_t*)malloc(input->dataLength * sizeof(uint8_t));
    output->dataLength = input->dataLength;
    output->dims = (uint64_t*)malloc(input->dimsLength * sizeof(uint64_t));
    output->dimsLength = input->dimsLength;
    memcpy((void*)output->dims, (void*)input->dims, input->dimsLength * sizeof(uint64_t));
    output->precision = input->precision;

    for (uint64_t i = 0; i < output->dataLength; i += sizeof(float)) {
        *(float*)(output->data + i) = *(float*)(input->data + i) + addValue - subValue;
    }

    fflush(stdout);
    return 0;
}

extern "C" int releaseBuffer(struct CustomNodeTensor* output) {
    printf("CUSTOM ADD_SUB NODE => Releasing Buffer (output name: %s)\n", output->name);
    fflush(stdout);
    free(output->data);
    free(output->dims);
    return 0;
}

extern "C" int releaseTensors(struct CustomNodeTensor* outputs) {
    printf("CUSTOM ADD_SUB NODE => Releasing Tensors\n");
    fflush(stdout);
    free(outputs);
    return 0;
}
